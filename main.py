from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util # type: ignore
import torch
from typing import List

app = FastAPI()

modelName = "drmzperx/inci-all-MiniLM-L6-v3"
model = SentenceTransformer(modelName)

class Query(BaseModel):
    query: str
    corpus: List[str]


@app.get("/")
def read_root():
    return {"Skinlyzer": "Similarity API v1"}


@app.post("/similarity/{domain}")
def read_item(domain: str, query: Query):
    print("Model: " + modelName)
    print("Domain: " + domain)

    # queryIn = "What is the man eating?"

    # corpus = ['A man is eating food.',
    #       'A man is eating a piece of bread.',
    #       'The girl is carrying a baby.',
    #       'A man is riding a horse.',
    #       'A woman is playing violin.',
    #       'Two men pushed carts through the woods.',
    #       'A man is riding a white horse on an enclosed ground.',
    #       'A monkey is playing drums.',
    #       'A cheetah is running behind its prey.'
    #       ]
    corpus = query.corpus
    
    queryIn = query.query
    print("Query: " + queryIn)
    print("Corpus: " + str(corpus))

    query_embedding = model.encode(queryIn)
    corpus_embeddings = model.encode(corpus)

    dot_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(dot_scores, k=len(corpus))

    list_results = []
    for score, idx in zip(top_results[0], top_results[1]):
        list_results.append({"text": corpus[idx], "sim": "{:.4f}".format(score)})
        print(corpus[idx], "(Score: {:.4f})".format(score))

    return list_results


@app.get("/test/{text}")
def update_item(text: str):
    return {"text": text}
