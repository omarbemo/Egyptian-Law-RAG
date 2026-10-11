from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import json
import numpy as np


#Embedding for DENSE vector

dense_model = SentenceTransformer("BAAI/bge-m3", device="cpu")

with open("articles/civil_code.json", encoding="utf-8") as f:
    records = json.load(f)

texts_en = [r["text_en"] for r in records]
texts_ar = [r["text_ar"] for r in records]
embeddings_en = dense_model.encode(texts_en, show_progress_bar=True)
embeddings_ar = dense_model.encode(texts_ar, show_progress_bar=True)


np.save("articles/embeddings_en.npy", embeddings_en)
np.save("articles/embeddings_ar.npy", embeddings_ar)


