from numba import njit
import numpy as np
from heapq import heappush, heapreplace 
from typing import List



class SimilarityNode(object):
    """
    Class to represent a node in the minheap to optimize the memory for the std python Tuple.
    """
    __slots__ = ('_similarity', '_idx')

    def __init__(self, similarity: np.double, idx: int):
        self._similarity = similarity
        self._idx = idx

    def __lt__(self, other):
        return self._similarity < other._similarity
    
    def __repr__(self) -> str:
        return f"(s:{self._similarity}, idx: {self._idx})\n"


class EmbeddingWithSimilarity(object):
    """
    Class that contains the embedding with its similarity.
    """
    __slots__ = ('_embedding', '_similarity')

    def __init__(self, embedding: np.ndarray[np.double], similarity: np.double) -> None:
        self._embedding = embedding
        self._similarity = similarity

    def __repr__(self) -> str:
        return f"(e:{self._embedding}, s: {self._similarity})\n"


@njit
def _calculate_similarities(
        target: np.ndarray[np.double],
        vembeddings: np.ndarray[np.ndarray[np.double]]
    ) -> np.ndarray[np.double]:
    """
    Given a target, figures out all similarities for a given set of embeddings.

    Time complexity: O(len(vembeddings))
    Space complexity: O(len(vembeddings))

    :param target: given target.
    :param vembeddings: set of embeddings.
    :return: An array of similarities for the given set of embeddings.
    """
    vlen = len(vembeddings)
    # vector to store the similarities.
    similarities = np.zeros(vlen, dtype=np.double)
    # pre-compute reiterative values to avoid repeating operations.
    sum_vector_squared = np.sum(target ** 2) ** 0.5
    sum_embeddings_squared = np.sum(vembeddings ** 2, axis=1) ** 0.5
    for i in range(0, vlen):
        similarities[i] = (np.sum(target * vembeddings[i])) / (sum_vector_squared * sum_embeddings_squared[i])
    return similarities



def cosine_similarity_search(
    target: np.ndarray[np.double],
    top_k: int,
    vembeddings: np.ndarray[np.ndarray[np.double]]
    ) -> List[EmbeddingWithSimilarity]:
    """
    The function performs a cosine similarity search that retrieves the 
    top k embeddings given a target embedding and a vector of multiple
    embeddings.

    :param target: target embedding
    :param top_k: k elements to retrieve.
    :param vembeddings: vector of multiple embeddings.

    :return: a List[EmbeddingWithSimilarity].
    """

    vlen: int = len(vembeddings)
    
    if vlen < top_k:
        raise ValueError("[ERROR]: top_k parameter is greater than embeddings' length.")
    
    # In order to retrieve top k elements we use a min heap of k elements.
    minheap: List[SimilarityNode] = []
    
    similarities: np.ndarray[np.double] = _calculate_similarities(target, vembeddings)
    # fill the min heap with the first k similarities.
    for i in range(0, top_k):
        heappush(minheap, SimilarityNode(similarities[i], i))
    
    # loop over the remaining similarities from [k:] and 
    # update the min heap of k elements.
    for i in range(top_k, vlen):
        # if the current similarity is greater than the lowest
        # similarity in the heap, remove the heap's top and add
        # the similarity.
        if similarities[i] > minheap[0]._similarity:
            heapreplace(minheap, SimilarityNode(similarities[i],i))
    
    # Once k embeddings are sorted in ascendent way, retrieve the embeddings 
    # along their similarity scores.
    embeddings: List[EmbeddingWithSimilarity] = []
    for i in range(0, top_k):
        embeddings.append(EmbeddingWithSimilarity(
            vembeddings[minheap[i]._idx],
            minheap[i]._similarity    
            )
        )
    return embeddings
