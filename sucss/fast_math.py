from numba import njit
import numpy as np


@njit
def cosine_similarity_search(vector: np.ndarray, top_k: int, embeddings: np.ndarray):
    """
    Given a question converted into an embedding of size (1536,)

    :param vector: query embedding.
    :param top_k: retrieves the top 5 embeddings.
    :param embeddings: embeddings vectors that shapes the vector database.
    :return: a tuple containing (indices: list of best top k indices, best_data: cosine similarity)
    """
    best_data = np.zeros(top_k, dtype=np.double)
    similarity = np.zeros(len(embeddings), dtype=np.double)
    sum_vector_squared = np.sum(vector ** 2) ** 0.5
    sum_embeddings_squared = np.sum(embeddings ** 2, axis=1) ** 0.5
    for i in range(0, len(embeddings)):
        similarity[i] = (np.sum(vector * embeddings[i])) / (sum_vector_squared * sum_embeddings_squared[i])
    #print(similarity)
    indices = np.argpartition(similarity, -top_k)[-top_k:]
    for i in range(0, top_k):
        best_data[i] = similarity[indices[i]]
    #print(indices)
    #print(best_data)
    return indices, best_data
