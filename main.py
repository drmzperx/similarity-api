import os
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedder import Embedder

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelName = "incitrainer/inci-w2v-embedder"

embedder = Embedder(
    embeddings_path=os.path.join(BASE_DIR, "models", "inci_embeddings.pt"),
    regression_path=os.path.join(BASE_DIR, "models", "inci_regression.pt"),
    synos_path=os.path.join(BASE_DIR, "models", "inci_synos.jsonl"),
)


class Query(BaseModel):
    list_a: str
    list_b: str


@app.get("/")
def read_root():
    return {"Skinlyzer": "Similarity API v2"}


@app.post("/similarity/{domain}")
def similarity(domain: str, query: Query):
    print("Model: " + modelName)
    print("Domain: " + domain)

    try:
        result = embedder.compare(query.list_a, query.list_b)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@app.get("/status")
def status():
    return {"status": "ok", "model": modelName}


@app.get("/test/{text}")
def echo(text: str):
    return {"text": text}
