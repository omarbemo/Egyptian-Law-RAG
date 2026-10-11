from fastembed import SparseTextEmbedding
import numpy as np
import json
import gc
import os

splade_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
with open("civil_code.json", encoding="utf-8") as f:
    records = json.load(f)

texts_en = [r["text_en"] for r in records]
texts_ar = [r["text_ar"] for r in records]

def embed_sparse_to_npy(model, texts, out_path, batch_size=50):
    all_data = []
    temp_chunks = []
    base_dir = os.path.dirname(out_path)
 
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = list(model.embed(batch))
        
        # Process and save this batch to a temporary file
        batch_data = np.array([(sv.indices, sv.values) for sv in vectors], dtype=object)
        chunk_path = f"{out_path}.chunk_{i}.npy"
        np.save(chunk_path, batch_data, allow_pickle=True)
        temp_chunks.append(chunk_path)
        
        print(f"{out_path}: {i + len(batch)}/{len(texts)}")
        
        # Explicitly free memory
        del vectors, batch_data, batch
        gc.collect()
    
    # Load and merge all chunks into one file
    merged_data = []
    for chunk_path in temp_chunks:
        merged_data.extend(np.load(chunk_path, allow_pickle=True))
        os.remove(chunk_path)
        
    np.save(out_path, np.array(merged_data, dtype=object), allow_pickle=True)
    del merged_data
    gc.collect()
    print(f"Finished saving {out_path} successfully!")

embed_sparse_to_npy(splade_model, texts_en, "/articles/sparse_en.npy", batch_size=25)
embed_sparse_to_npy(bm25_model, texts_ar, "articles/sparse_ar.npy", batch_size=50)