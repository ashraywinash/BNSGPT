from chromadb import Documents, EmbeddingFunction, Embeddings
import numpy as np
from typing import List, Union


class FastTextEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        # Ensure input is a list of documents
        if isinstance(input, str):
            input = [input]
        
        # Generate embeddings for each document
        embeddings = [self._get_embedding(text) for text in input]
        return embeddings

    def _get_embedding(self, text: str) -> List[float]:
        # Get the embedding vector from the model
        embedding = self.model.get_sentence_vector(text)
        
        # Convert NumPy array to a list of floats
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()  # Convert to list of floats
        
        # Ensure the embedding is a list of floats
        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
            return embedding
        else:
            raise ValueError("Embedding conversion failed. Ensure the embedding is a list of floats.")


