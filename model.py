import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

DATA_PATH = "scriptures/Bhagwad_Gita.csv"
EMB_PATH = "models/gita_embeddings.npy"
INDEX_PATH = "models/gita_faiss.index"

df = pd.read_csv(DATA_PATH).dropna(subset=['EngMeaning']).reset_index(drop=True)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

if not os.path.exists(EMB_PATH):
    print("Creating embeddings...")
    embeddings = embed_model.encode(df['EngMeaning'].tolist(), show_progress_bar=True)
    np.save(EMB_PATH, embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    print("Saved embeddings + index.")
else:
    print("Embeddings already exist.")