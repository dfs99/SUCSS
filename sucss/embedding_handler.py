import numpy as np
from typing import Callable, Dict, Any
import json


class EmbeddingHandler:
    """
    A class that handles embeddings.

    It's possible to load from disk data or create random data.

    A class that creates a 'num_embeddings' random embeddings of
    dimension 'dimension' through numpy.

    It can take a function to create specific metadata.
    The function itself might have any args but must return a single dict for metadata.
    """

    def __init__(
            self,
            dimension: int,
            num_embeddings: int,
            last_id: int = None,
            metadata_func: Callable[[Any], Dict] = None,
            from_disk: bool = False
    ):
        self._dimension = dimension
        self._num_embeddings = num_embeddings
        if from_disk: # if we want to load data from disk, create a zeros memory layout first.
            self._embeddings = np.zeros((self._num_embeddings, self._dimension), dtype=np.double)
        else:
            self._embeddings = np.random.random((self._num_embeddings, self._dimension))
        # to be able to keep a unique identifier for each embedding.
        if from_disk is False:
            if last_id is None:
                self._ids = np.arange(0, self._num_embeddings, 1, dtype=int)
            else:
                self._ids = np.arange(last_id, last_id + self._num_embeddings, 1, dtype=int)
        else:
            self._ids = np.zeros(self._num_embeddings, dtype=int)
        self._metadata_func = metadata_func

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def dimension(self):
        return self._dimension

    def to_disk(self, file_path):
        with open(file_path, "w", encoding='utf-8') as file:
            current_embeddings = {
                "handler_metadata": {
                    "dimension": self._dimension,
                    "num_embeddings": self._num_embeddings
                },
                "embeddings": []
            }
            for i in range(0, self._num_embeddings):
                current_embeddings['embeddings'].append(
                    {
                        "id": int(self._ids[i]),
                        "metadata": {} if self._metadata_func is None else self._metadata_func(),
                        "content": self._embeddings[i].tolist()
                    }
                )
            json.dump(current_embeddings, file, indent=3)

    def __str__(self):
        return f"Embedding Handler:\nDimension: {self._dimension}\nNum embeddings: {self._num_embeddings}\n"

    @classmethod
    def from_disk(cls, path_file, metadata_func: Callable[[Any], Dict] = None):
        with open(path_file, "r", encoding='utf-8') as file:
            json_data = json.load(file)
        dimension = json_data['handler_metadata']['dimension']
        num_embeddings = json_data['handler_metadata']['num_embeddings']
        my_class = cls(dimension, num_embeddings, metadata_func=metadata_func, from_disk=True)
        # substitute vector and idx data.
        for i in range(0, len(json_data['embeddings'])):
            my_class._ids[i] = json_data['embeddings'][i]['id']
            for j in range(0, dimension):
                my_class._embeddings[i][j] = json_data['embeddings'][i]['content'][j]
        return my_class


if __name__ == "__main__":
    ec = EmbeddingHandler(10, 5)
    print(ec)
    #ec.to_disk("test.json")
    e2 = EmbeddingHandler.from_disk("test.json")