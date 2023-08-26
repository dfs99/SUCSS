from sucss import EmbeddingHandler, cosine_similarity_search
from timeit import default_timer as timer
import numpy as np


if __name__ == "__main__":

    OPENAI_EMBEDDING_SIZE = 1536
    NUMBER = 100000
    TOP_K = 3
    query = np.random.random((OPENAI_EMBEDDING_SIZE,))
    ehandler = EmbeddingHandler(OPENAI_EMBEDDING_SIZE, NUMBER)
    
    start = timer()
    idx = cosine_similarity_search(query, TOP_K, ehandler.embeddings)
    print(f"Number instances: {NUMBER} Elapsed time: {timer()-start}")
    print(f"top: {TOP_K} embeddings with their similarities:")
    print(idx)
