import sucss
from timeit import default_timer as timer
import numpy as np


if __name__ == "__main__":

    OPENAI_EMBEDDING_SIZE = 1536
    NUMBER = 10000
    TOP_K = 5
    query = np.random.random((OPENAI_EMBEDDING_SIZE,))

    ehandler = sucss.EmbeddingHandler(OPENAI_EMBEDDING_SIZE, NUMBER)

    start = timer()
    idx, sc = sucss.cosine_similarity_search(query, TOP_K, ehandler.embeddings)
    print(f"Elapsed time: {timer()-start}")
