from fastembed.image import image_embedding_base
from fastembed.image import image_embedding_base
from fastembed.image import image_embedding_base
from huggingface_hub.utils import endpoint_helpers
from huggingface_hub.utils import endpoint_helpers
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv() 

dense_model = SentenceTransformer("BAAI/bge-m3", device="cpu") #general dense
splade_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")  # English sparse
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")                   # Arabic sparse

collection_name = "civil_code"

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["api_key"],
    timeout=120,
    prefer_grpc=True,
)

def hybrid_search(query, lang="en", limit=5):
    query_dense = dense_model.encode(query).tolist()

    if lang == "en":
        sparse_vec = next(splade_model.embed([query]))
        dense_name, sparse_name = "en_dense", "en_sparse"
    else:
        sparse_vec = next(bm25_model.embed([query]))
        dense_name, sparse_name = "ar_dense", "ar_sparse"

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=query_dense, using=dense_name, limit=20),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using=sparse_name,
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    return results.points

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask(query, lang ,top_k=5):
    points = hybrid_search(query, lang=lang, limit=top_k)
    text_key = "text_en" if lang == "en" else "text_ar"
    ids = [p.payload["article_number"] for p in points]
    contexts = [p.payload[text_key] for p in points]

    context_block = "\n\n".join(f"Article {i}: {t}" for i, t in zip(ids, contexts))

    prompt = f"""Answer the question using only the articles below. Cite the article number(s) you used.

    {context_block}

    Question: {query}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content , contexts , ids
question ="ما هو القانون الذي ينظم دفاتر المواليد و التبليغات المتعلقه مبها"
print(ask(question , "ar", 1))