from groq import Groq
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import json
import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()
client = os.getenv("GROQ_API_KEY")


model = SentenceTransformer("BAAI/bge-m3",device="cpu")
embeddings_en = np.load("articles/embeddings_en.npy")

with open("articles/civil_code.json", encoding="utf-8") as f:
    records = json.load(f)

def search(query, top_k=5):
    query_vec = model.encode(query)
    sims = util.cos_sim(query_vec, embeddings_en)[0]
    top_idx = sims.argsort(descending=True)[:top_k]
    return [(records[i]["article_number"], records[i]["text_en"], sims[i]) for i in top_idx]

def ask(query, top_k=5):
    results = search(query, top_k=top_k)

    context = "\n\n".join(
        f"Article {num}: {text}" for num, text, score in results
    )

    prompt = f"""Answer the question using only the articles below. Cite the article number(s) you used.

{context}

Question: {query}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
question1="solely of giving a creditor an\nunjustified preference over another creditor only entails\nthe?"
question2="If I am a victim of fraudulent misrepresentation by a third party, can I automatically demand the contract be avoided?"
question3="What is the legal distinction between things that are outside the ambit of trade by their ""nature versus those outside it by law"
question4="Does a declaration of intention become completely ineffective if the person making it becomes legally incapable before the intended recipient is made aware of it?"
questions = [question1,question2,question3,question4]
for q in questions:
    print(q, "->", ask(q))