from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
from typing import Any, Optional, List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

model = TextEmbedding("BAAI/bge-small-en-v1.5")

class EmbedRequest(BaseModel):
    text: str

class EmbedBatchRequest(BaseModel):
    texts: List[str]

def flatten_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, (int, float, bool)):
        return str(value)
    elif isinstance(value, list):
        return " ".join(flatten_value(v) for v in value)
    elif isinstance(value, dict):
        return " ".join(flatten_value(v) for v in value.values())
    else:
        return ""

def object_to_text(data: dict, fields: Optional[List[str]]) -> str:
    subset = {k: v for k, v in data.items() if k in fields} if fields else data
    return " ".join(
        f"{k}: {flatten_value(v)}"
        for k, v in subset.items()
        if flatten_value(v)
    )

def embed_text(text: str) -> List[float]:
    return list(model.embed([text]))[0].tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    return [v.tolist() for v in model.embed(texts)]

@app.post("/embed")
def embed(body: EmbedRequest):
    vector = embed_text(body.text)
    return {
        "text": body.text,
        "embedding": vector,
        "dimensions": len(vector)
    }

@app.post("/embed/object")
def embed_object(body: Dict[str, Any]):
    """Accepts both {data: {...}} and raw {...}"""
    if "data" in body and isinstance(body["data"], dict):
        # wrapped format: { "data": {...}, "fields": [...] }
        data = body["data"]
        fields = body.get("fields", None)
    else:
        # raw format: { "key": "value", ... }
        fields = body.pop("fields", None)
        data = body

    text = object_to_text(data, fields)
    vector = embed_text(text)
    return {
        "text": text,
        "embedding": vector,
        "dimensions": len(vector)
    }

@app.post("/embed/batch")
def embed_batch(body: EmbedBatchRequest):
    vectors = embed_texts(body.texts)
    return {
        "embeddings": [
            {"text": t, "embedding": v, "dimensions": len(v)}
            for t, v in zip(body.texts, vectors)
        ]
    }