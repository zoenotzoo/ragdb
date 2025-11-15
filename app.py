"""FastAPI server that implements a minimal Chroma-like vector database.

The implementation focuses on the core workflow required for a Retrieval
Augmented Generation (RAG) system:

* collection-level management of embeddings and metadata
* approximate nearest neighbour search powered by FAISS
* metadata based filtering
* simple REST API surface for upsert/query/delete operations

The goal is to provide a small yet functional reference that can be extended
towards a production ready service.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

import faiss  # type: ignore
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator
from sentence_transformers import SentenceTransformer


DB_PATH = Path("meta.db")
INDEX_DIR = Path("indices")
ID_MAP_SUFFIX = "_ids.json"

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def ensure_storage() -> None:
    """Create on-disk folders / tables that are required by the service."""

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS __collections__(
                name TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL
            );
            """
        )
        conn.commit()


MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""

    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return MODEL_CACHE[model_name]


class UpsertBody(BaseModel):
    ids: List[str]
    documents: Optional[List[Optional[str]]] = None
    embeddings: Optional[List[List[float]]] = None
    metadatas: Optional[List[Optional[Dict[str, object]]]] = None

    @root_validator
    def validate_lengths(cls, values: Dict[str, object]) -> Dict[str, object]:
        ids = values.get("ids") or []
        length = len(ids)
        for key in ("documents", "embeddings", "metadatas"):
            payload: Optional[List[object]] = values.get(key)  # type: ignore[assignment]
            if payload is not None and len(payload) != length:
                raise ValueError(f"`{key}` must have the same length as `ids`")
        documents = values.get("documents")
        embeddings = values.get("embeddings")
        if documents is None and embeddings is None:
            raise ValueError("Either `documents` or `embeddings` must be provided")
        return values


class QueryBody(BaseModel):
    query_texts: Optional[List[str]] = Field(default=None)
    query_embeddings: Optional[List[List[float]]] = Field(default=None)
    n_results: int = Field(default=5, ge=1)
    where: Optional[Dict[str, object]] = Field(default=None)

    @root_validator
    def validate_query_payload(cls, values: Dict[str, object]) -> Dict[str, object]:
        if values.get("query_texts") is None and values.get("query_embeddings") is None:
            raise ValueError("Either `query_texts` or `query_embeddings` must be provided")
        return values


class DeleteBody(BaseModel):
    ids: Optional[List[str]] = None
    where: Optional[Dict[str, object]] = None

    @root_validator
    def validate_deletion_payload(cls, values: Dict[str, object]) -> Dict[str, object]:
        if values.get("ids") is None and values.get("where") is None:
            raise ValueError("Either `ids` or `where` must be provided")
        return values


class Collection:
    """Represents a logical namespace that stores embeddings and metadata."""

    def __init__(self, name: str, model_name: str = DEFAULT_MODEL) -> None:
        ensure_storage()
        self.name = name
        self.model_name = model_name
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = Lock()

        self.model = get_model(model_name)
        self.dimension = self._ensure_collection_record()

        self.table = f"collection__{self.name}"
        self._create_table()

        self.index_path = INDEX_DIR / f"{self.name}.faiss"
        self.id_map_path = INDEX_DIR / f"{self.name}{ID_MAP_SUFFIX}"
        self.index = faiss.IndexFlatIP(self.dimension)
        self._entries: List[Dict[str, object]] = []
        self._id_map: List[str] = []
        self._rebuild_index()

    def _ensure_collection_record(self) -> int:
        cur = self.conn.execute(
            "SELECT dimension, model_name FROM __collections__ WHERE name = ?", (self.name,)
        )
        row = cur.fetchone()
        if row is not None:
            if row["model_name"] != self.model_name:
                raise HTTPException(
                    status_code=400,
                    detail="Collection already exists with a different model",
                )
            return int(row["dimension"])

        dimension = int(self.model.get_sentence_embedding_dimension())
        self.conn.execute(
            "INSERT INTO __collections__(name, model_name, dimension) VALUES(?,?,?)",
            (self.name, self.model_name, dimension),
        )
        self.conn.commit()
        return dimension

    def _create_table(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id TEXT PRIMARY KEY,
                document TEXT,
                metadata TEXT,
                embedding BLOB,
                deleted INTEGER DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def _fetch_existing(self, ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
        ids = list(ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = (
            f"SELECT id, document, metadata FROM {self.table} "
            f"WHERE id IN ({placeholders})"
        )
        rows = self.conn.execute(query, ids).fetchall()
        existing: Dict[str, Dict[str, object]] = {}
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            existing[row["id"]] = {
                "document": row["document"],
                "metadata": metadata,
            }
        return existing

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings
        faiss.normalize_L2(embeddings)
        return embeddings

    def upsert(
        self,
        ids: List[str],
        documents: Optional[List[Optional[str]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Optional[Dict[str, object]]]] = None,
    ) -> None:
        with self._lock:
            existing = self._fetch_existing(ids)

            docs_to_store: List[str] = []
            metadata_to_store: List[str] = []
            embeddings_to_store: List[np.ndarray] = []

            if embeddings is not None:
                raw_embeddings = np.asarray(embeddings, dtype="float32")
                if raw_embeddings.ndim != 2 or raw_embeddings.shape[1] != self.dimension:
                    raise HTTPException(
                        status_code=400,
                        detail="Embeddings do not match collection dimensionality",
                    )
                normalized_embeddings = self._normalize_embeddings(raw_embeddings.copy())
            else:
                normalized_embeddings = None

            for idx, identifier in enumerate(ids):
                existing_entry = existing.get(identifier, {})

                document = (
                    documents[idx]
                    if documents is not None and documents[idx] is not None
                    else existing_entry.get("document")
                )
                metadata = (
                    metadatas[idx]
                    if metadatas is not None and metadatas[idx] is not None
                    else existing_entry.get("metadata")
                )

                if document is None and normalized_embeddings is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Document text required when embeddings are not provided",
                    )

                if normalized_embeddings is not None:
                    vector = normalized_embeddings[idx]
                else:
                    assert documents is not None  # for type checkers
                    encoded = self.model.encode(
                        [document or ""],
                        normalize_embeddings=True,
                    )
                    vector = encoded[0]

                docs_to_store.append(document or "")
                metadata_to_store.append(json.dumps(metadata or {}))
                embeddings_to_store.append(np.asarray(vector, dtype="float32"))

            for i, identifier in enumerate(ids):
                self.conn.execute(
                    f"""
                    INSERT INTO {self.table}(id, document, metadata, embedding, deleted, updated_at)
                    VALUES(?,?,?,?,0,CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        document=excluded.document,
                        metadata=excluded.metadata,
                        embedding=excluded.embedding,
                        deleted=0,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        identifier,
                        docs_to_store[i],
                        metadata_to_store[i],
                        embeddings_to_store[i].tobytes(),
                    ),
                )

            self.conn.commit()
            self._rebuild_index()

    def query(
        self,
        query_texts: Optional[List[str]],
        query_embeddings: Optional[List[List[float]]],
        n_results: int,
        where: Optional[Dict[str, object]],
    ) -> List[List[Dict[str, object]]]:
        with self._lock:
            if self.index.ntotal == 0:
                return [[] for _ in range(len(query_texts or query_embeddings or []))]

            if query_embeddings is not None:
                queries = np.asarray(query_embeddings, dtype="float32")
                if queries.ndim != 2 or queries.shape[1] != self.dimension:
                    raise HTTPException(
                        status_code=400,
                        detail="Query embeddings do not match collection dimensionality",
                    )
                queries = self._normalize_embeddings(queries.copy())
            else:
                assert query_texts is not None
                queries = self.model.encode(
                    query_texts,
                    normalize_embeddings=True,
                ).astype("float32")

            n_results = min(n_results, self.index.ntotal)
            distances, indices = self.index.search(queries, n_results)

            results: List[List[Dict[str, object]]] = []
            for query_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                matches: List[Dict[str, object]] = []
                for rank, idx_value in enumerate(idx_row):
                    if idx_value < 0 or idx_value >= len(self._entries):
                        continue
                    entry = self._entries[idx_value]
                    if where and not self._metadata_matches(entry["metadata"], where):
                        continue
                    matches.append(
                        {
                            "id": entry["id"],
                            "document": entry["document"],
                            "metadata": entry["metadata"],
                            "score": float(dist_row[rank]),
                        }
                    )
                results.append(matches)
            return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, object]] = None,
    ) -> int:
        with self._lock:
            if ids:
                placeholders = ",".join("?" for _ in ids)
                query = f"UPDATE {self.table} SET deleted=1 WHERE id IN ({placeholders})"
                cursor = self.conn.execute(query, ids)
                affected = cursor.rowcount
            elif where:
                to_delete = [entry["id"] for entry in self._entries if self._metadata_matches(entry["metadata"], where)]
                if not to_delete:
                    return 0
                placeholders = ",".join("?" for _ in to_delete)
                query = f"UPDATE {self.table} SET deleted=1 WHERE id IN ({placeholders})"
                cursor = self.conn.execute(query, to_delete)
                affected = cursor.rowcount
            else:
                affected = 0

            self.conn.commit()
            self._rebuild_index()
            return affected

    def info(self) -> Dict[str, object]:
        cur = self.conn.execute(
            f"SELECT COUNT(*) as total FROM {self.table} WHERE deleted=0"
        )
        total = cur.fetchone()["total"]
        return {
            "name": self.name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "entries": total,
        }

    def drop(self) -> None:
        with self._lock:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
            self.conn.execute(
                "DELETE FROM __collections__ WHERE name = ?",
                (self.name,),
            )
            self.conn.commit()
            if self.index_path.exists():
                self.index_path.unlink()
            if self.id_map_path.exists():
                self.id_map_path.unlink()
            self._entries.clear()
            self._id_map.clear()
            self.index.reset()

    def _metadata_matches(self, metadata: Dict[str, object], where: Dict[str, object]) -> bool:
        for key, expected in where.items():
            if metadata.get(key) != expected:
                return False
        return True

    def _rebuild_index(self) -> None:
        cursor = self.conn.execute(
            f"SELECT id, document, metadata, embedding FROM {self.table} WHERE deleted=0"
        )
        rows = cursor.fetchall()

        vectors: List[np.ndarray] = []
        entries: List[Dict[str, object]] = []
        for row in rows:
            if row["embedding"] is None:
                continue
            vector = np.frombuffer(row["embedding"], dtype="float32")
            if vector.size != self.dimension:
                continue
            vectors.append(vector)
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            entries.append(
                {
                    "id": row["id"],
                    "document": row["document"],
                    "metadata": metadata,
                }
            )

        matrix: Optional[np.ndarray] = None
        if vectors:
            matrix = np.vstack(vectors).astype("float32")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(matrix)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        self._entries = entries
        self._id_map = [entry["id"] for entry in entries]
        self._persist_index()

    def _persist_index(self) -> None:
        if self.index.ntotal > 0:
            faiss.write_index(self.index, str(self.index_path))
            with self.id_map_path.open("w", encoding="utf-8") as fh:
                json.dump(self._id_map, fh)
        else:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.id_map_path.exists():
                self.id_map_path.unlink()


class CollectionManager:
    """Lifecycle manager that caches `Collection` instances in memory."""

    def __init__(self) -> None:
        self._collections: Dict[str, Collection] = {}
        self._lock = Lock()

    def get(self, name: str, model_name: str = DEFAULT_MODEL) -> Collection:
        with self._lock:
            collection = self._collections.get(name)
            if collection is None:
                collection = Collection(name, model_name=model_name)
                self._collections[name] = collection
            return collection

    def drop(self, name: str) -> None:
        with self._lock:
            collection = self._collections.pop(name, None)
            if collection is None:
                collection = Collection(name)
            collection.drop()


app = FastAPI(title="Mini Vector DB", version="0.1.0")
manager = CollectionManager()


class CollectionCreateBody(BaseModel):
    name: str
    model_name: Optional[str] = Field(default=DEFAULT_MODEL)


@app.post("/collections")
def create_collection(payload: CollectionCreateBody) -> Dict[str, object]:
    collection = manager.get(payload.name, payload.model_name or DEFAULT_MODEL)
    return collection.info()


@app.get("/collections/{name}")
def get_collection_info(name: str) -> Dict[str, object]:
    collection = manager.get(name)
    return collection.info()


@app.delete("/collections/{name}")
def delete_collection(name: str) -> Dict[str, bool]:
    manager.drop(name)
    return {"ok": True}


@app.post("/collections/{name}/upsert")
def upsert(name: str, body: UpsertBody) -> Dict[str, bool]:
    collection = manager.get(name)
    collection.upsert(
        ids=body.ids,
        documents=body.documents,
        embeddings=body.embeddings,
        metadatas=body.metadatas,
    )
    return {"ok": True}


@app.post("/collections/{name}/query")
def query(name: str, body: QueryBody) -> Dict[str, List[List[Dict[str, object]]]]:
    collection = manager.get(name)
    results = collection.query(
        query_texts=body.query_texts,
        query_embeddings=body.query_embeddings,
        n_results=body.n_results,
        where=body.where,
    )
    return {"results": results}


@app.post("/collections/{name}/delete")
def delete(name: str, body: DeleteBody) -> Dict[str, object]:
    collection = manager.get(name)
    affected = collection.delete(ids=body.ids, where=body.where)
    return {"deleted": affected}


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


__all__ = [
    "app",
]

