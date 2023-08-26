# SUCSS: Speeding up Cosine Similarity Search

Even though Vector Databases do perform Cosine Similarity Search way 
faster due to the use of more optimized algorithms. This repository
contains a fast way to compute cosine similarity search using plain
python.

The approach is the following, once all the similarities have been
figured out, a binary min heap of size k is created to retrieve the 
top k embeddings with higher similarity score. 

Note that the array retrieved of size k is not sorted in ascending 
way but in a min binary heap shape since the heap is stored inplace.
