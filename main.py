from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util # type: ignore
import torch
from typing import List

app = FastAPI()

modelName = "drmzperx/inci-all-MiniLM-L6-v4"
model = SentenceTransformer(modelName)

class Query(BaseModel):
    query: str
    corpus: List[str]
    authid: List[str]

@app.get("/")
def read_root():
    return {"Skinlyzer": "Similarity API v1"}


@app.post("/similarity/{domain}")
def read_item(domain: str, query: Query | None = None):
    print("Model: " + modelName)
    print("Domain: " + domain)
    # print("Body: " + str(query))

    corpus = query.corpus
    authids = query.authid
    
    queryIn = query.query
    # print("Query: " + queryIn)
    # print("Corpus: " + str(corpus))

    query_embedding = model.encode(queryIn)
    corpus_embeddings = model.encode(corpus)

    dot_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(dot_scores, k=len(corpus))

    list_results = []
    for score, idx in zip(top_results[0], top_results[1]):
        list_results.append({"text": corpus[idx], "sim": "{:.4f}".format(score)})
        print(authids[idx], "(Score: {:.4f})".format(score))

    return list_results

@app.get("/status")
def update_item():
    return {"status": "ok", "model": modelName}

@app.get("/test/{text}")
def update_item(text: str):
    return {"text": text}
