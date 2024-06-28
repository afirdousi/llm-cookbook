import os
import faiss
import time
import numpy as np
from tqdm import tqdm
from lamini.api.embedding import Embedding
from lamini import Lamini

from directory_helper import DirectoryLoader

# Define your API key (Bad practice - Never do this in prod)
API_KEY = 'YOUR_LAMINI_KEY_GOES_HERE'


class LaminiIndex:
    def __init__(self, loader):
        self.loader = loader
        self.build_index()

    def build_index(self):
        self.content_chunks = []
        self.index = None
        for chunk_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(chunk_batch)
            if embeddings is None or len(embeddings) == 0:
                print("No embeddings generated for chunk batch")
                continue
            if self.index is None:
                print(f"Initializing index with dimension {len(embeddings[0])}")
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            print(f"Adding {len(embeddings)} embeddings to the index")
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)
        if self.index is None:
            print("Warning: Index was not initialized. Check if embeddings were generated.")

    def get_embeddings(self, examples):
        ebd = Embedding(api_key=API_KEY)  # Pass the API key here
        embeddings = ebd.generate(examples)
        embedding_list = [embedding[0] for embedding in embeddings]
        return np.array(embedding_list)

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]

class QueryEngine:
    def __init__(self, index, k=5):
        self.index = index
        self.k = k
        self.model = Lamini(model_name="mistralai/Mistral-7B-Instruct-v0.1",
                            api_key=API_KEY)  # Pass the API key here)

    def answer_question(self, question):
        most_similar = self.index.query(question, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        print("------------------------------ Prompt ------------------------------\n" + prompt + "\n----------------------------- End Prompt -----------------------------")
        return self.model.generate("<s>[INST]" + prompt + "[/INST]")

class RetrievalAugmentedRunner:
    def __init__(self, dir, k=5):
        self.k = k
        self.loader = DirectoryLoader(dir)

    def train(self):
        self.index = LaminiIndex(self.loader)

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k)
        return query_engine.answer_question(query)

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    model = RetrievalAugmentedRunner(dir=data_dir)
    start = time.time()
    model.train()
    print("Time taken to build index: ", time.time() - start)
    while True:
        prompt = input("\n\nEnter another investment question (e.g. Have we invested in any generative AI companies in 2023?): ")
        start = time.time()
        print(model(prompt))
        print("\nTime taken: ", time.time() - start)

main()