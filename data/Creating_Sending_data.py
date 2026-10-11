import os
import json
import numpy as np
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="https://94a75d76-9df2-4109-a889-0631836b4535.eu-central-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2VkMTQzZGEtNGI2NS00ZGYzLTg5NjAtNmFmNTc4OTkzY2NhIn0.zu8vTm0jo9H5HakYqFiG6F-IneukRpEukCkH6ekK4g8",
    timeout=120,          # default is far too short for big batches
    prefer_grpc=True,     # binary protocol, much smaller than JSON floats
)
collection_name = "civil_code"

# --- 1. load your existing data ---
with open("articles/civil_code.json", encoding="utf-8") as f:
    records = json.load(f)


embeddings_en = np.load("articles/embeddings_en.npy")
embeddings_ar = np.load("articles/embeddings_ar.npy")
sparse_en = np.load("articles/sparse_en.npy",allow_pickle=True)
sparse_ar = np.load("articles/sparse_ar.npy",allow_pickle=True)



# --- 2. recreate the collection fresh ---
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "en_dense": models.VectorParams(size=embeddings_en.shape[1], distance=models.Distance.COSINE),
        "ar_dense": models.VectorParams(size=embeddings_ar.shape[1], distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "en_sparse":models.SparseVectorParams(),
        "ar_sparse":models.SparseVectorParams(),
    },
)

# --- 3. create payload indexes before uploading ---
for field in ["book", "chapter", "section"]:
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

client.create_payload_index(
    collection_name=collection_name,
    field_name="article_number",
    field_schema=models.PayloadSchemaType.INTEGER,
)

# --- 4. build points ---
points = [
    models.PointStruct(
        id=rec["article_number"],
        vector={
            "en_dense": embeddings_en[i].tolist(),
            "ar_dense": embeddings_ar[i].tolist(),
            "en_sparse": models.SparseVector(
                indices=sparse_en[i][0].tolist(),
                values=sparse_en[i][1].tolist(),
            ),
            "ar_sparse": models.SparseVector(
                indices=sparse_ar[i][0].tolist(),
                values=sparse_ar[i][1].tolist(),
            ),
        },
        payload={
            "article_number": rec["article_number"],
            "book": rec.get("book"),
            "chapter": rec.get("chapter"),
            "section": rec.get("section"),
            "text_en": rec["text_en"],
            "text_ar": rec["text_ar"],
            "citation": rec.get("citation"),
        },
    )
    for i, rec in enumerate(records)
]

client.upload_points(
    collection_name=collection_name,
    points=points,
    batch_size=32,     # smaller requests; lower to 16 if it still times out
    parallel=1,        # keep at 1 on a slow connection
    max_retries=5,
    wait=True,
)
# --- 6. verify ---
count = client.get_collection(collection_name).points_count
print(f"uploaded {len(points)} points, collection reports {count}")
