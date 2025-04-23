from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import UploadFile, File
from PIL import Image
import io
import torch
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request


app = FastAPI()

templates = Jinja2Templates(directory="templates")

## general agent Q&A

class QARequest(BaseModel):
    question: str

with open("faq_contexts.json", "r") as f:
    faq_contexts = json.load(f)

vectorizer = TfidfVectorizer().fit(list(faq_contexts.values()))
faq_vectors = vectorizer.transform(faq_contexts.values())
faq_keys = list(faq_contexts.keys())

### testing diff models, large too slow for suboptimal improvement
# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")

@app.post("/qa")
def answer_question(request: QARequest):
    question_vec = vectorizer.transform([request.question])
    sims = cosine_similarity(question_vec, faq_vectors).flatten()
    best_idx = sims.argmax()
    context_text = list(faq_contexts.values())[best_idx]

    result = qa_pipeline({
        "question": request.question,
        "context": context_text
    })

    return {
        "answer": result["answer"],
        "score": result["score"],
        "matched_context": faq_keys[best_idx]
    }


## text-based product recommendation

with open("product_catalog.json", "r") as f:
    product_catalog = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

product_texts = [p["name"] + " " + p["description"] for p in product_catalog]
product_embeddings = embedding_model.encode(product_texts)

class RecRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend_products(request: RecRequest):
    query_embedding = embedding_model.encode([request.query])
    sims = cosine_similarity(query_embedding, product_embeddings).flatten()
    ranked = sorted(zip(product_catalog, sims), key=lambda x: x[1], reverse=True)[:3]
    return [
        {
            "name": item["name"],
            "description": item.get("description", "N/A"),
            "price": item["price"],
            "score": round(float(score), 3)
        } for item, score in ranked
    ]


# ## image search

# clip_model = SentenceTransformer("clip-ViT-B-32")

# product_image_embeddings = []
# for item in product_catalog:
#     img_path = item.get("image_path")
#     if img_path:
#         try:
#             img = Image.open(img_path).convert("RGB")
#             emb = clip_model.encode(img, convert_to_tensor=True)
#             product_image_embeddings.append((item, emb))
#         except Exception as e:
#             print(f"Failed to load image {img_path}: {e}")

# @app.post("/search-by-image")
# async def search_by_image(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     image_embedding = clip_model.encode(image, convert_to_tensor=True)

#     ## matches >=0.8 only
#     results = []
#     for (item, emb) in product_image_embeddings:
#         sim = torch.nn.functional.cosine_similarity(image_embedding, emb, dim=0)
#         sim_score = sim.item()
#         if sim_score >= 0.8:
#             results.append((item, sim_score))

#     # show atleast one if nothing matches well (the best one)
#     if not results:
#         fallback_scores = [
#             (item, torch.nn.functional.cosine_similarity(image_embedding, emb, dim=0).item())
#             for (item, emb) in product_image_embeddings
#         ]
#         results = sorted(fallback_scores, key=lambda x: x[1], reverse=True)[:1]

#     return [
#         {
#             "name": item["name"],
#             "price": item["price"],
#             "score": round(float(score), 3)
#         }
#         for item, score in sorted(results, key=lambda x: x[1], reverse=True)
#     ]

@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


