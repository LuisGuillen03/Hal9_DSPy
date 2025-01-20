import numpy as np
from typing import List, Optional, Union
from typing import Optional
import os
import faiss
from dspy import Retrieve, Prediction
from dsp.utils import dotdict

class CustomOpenAIVectorizer:
    """
    This vectorizer uses the OpenAI API through Hal9 proxy to convert texts to embeddings.
    Changing `model` is not recommended. 
    `api_key` should be passed as an argument or as an env variable (`OPENAI_API_KEY`).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",  # Default model
        embed_batch_size: int = 1024,  # Batch size for embeddings
        openai_client=None,  # Allow passing the custom OpenAI client (Hal9 proxy)
    ):
        """
        Initialize the CustomOpenAIVectorizer with the model and batch size.

        Args:
            model (str): The OpenAI model to use for embeddings.
            openai_client (Optional): Custom OpenAI client using Hal9 proxy.
        """
        self.model = model
        self.embed_batch_size = embed_batch_size

        # If no openai_client provided, raise an error
        if openai_client is None:
            raise ValueError("Must provide a valid openai_client (Hal9 proxy)")

        self.openai_client = openai_client

    def __call__(self, text_to_vectorize: List[str]) -> np.ndarray:
        """
        Generate embeddings for the input examples.

        Args:
            inp_examples (List[str]): List of input examples (texts).

        Returns:
            np.ndarray: Array of generated embeddings.
        """
        embeddings_list = []
        # Calculate the number of batches required
        n_batches = (len(text_to_vectorize) - 1) // self.embed_batch_size + 1

        # Process each batch
        for cur_batch_idx in range(n_batches):
            start_idx = cur_batch_idx * self.embed_batch_size
            end_idx = (cur_batch_idx + 1) * self.embed_batch_size
            cur_batch = text_to_vectorize[start_idx:end_idx]

            # Using Hal9-proxied OpenAI client to make the embeddings API call
            response = self.openai_client.embeddings.create(
                input=cur_batch,
                model=self.model
            )

            # Extract embeddings from the response (correcting access to embedding object)
            cur_batch_embeddings = [cur_obj.embedding for cur_obj in response.data]
            embeddings_list.extend(cur_batch_embeddings)

        # Convert the list of embeddings into a numpy array and return
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings
    
class Hal9_FaissRM(Retrieve):
    def __init__(self, 
                 document_chunks, 
                 vectorizer=None, 
                 k: int = 3, 
                 index_file="./.storage/.faiss_index_file.index", 
                 embeddings_file="./.storage/.faiss_embeddings.npy", 
                 update=False):
        """Initializes the Faiss retrieval module and optionally loads the Faiss index and embeddings from files.

        Args:
            update (bool): If True, will update the Faiss index with new document chunks and embeddings.
        """
        self.index_file = index_file
        self.embeddings_file = embeddings_file
        self.update = update  # Whether to update or recreate the index
        self._vectorizer = vectorizer

        # If we need to update the index, recreate everything
        if self.update or not os.path.exists(self.index_file) or not os.path.exists(self.embeddings_file):
            # Generate embeddings for the document chunks
            embeddings = self._vectorizer(document_chunks)
            xb = np.array(embeddings)
            d = len(xb[0])  # Dimension of the embeddings

            # Initialize Faiss index based on the number of document chunks
            if len(xb) < 100:
                self._faiss_index = faiss.IndexFlatL2(d)
                self._faiss_index.add(xb)
            else:
                nlist = 100
                quantizer = faiss.IndexFlatL2(d)
                self._faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                self._faiss_index.train(xb)
                self._faiss_index.add(xb)

            # Save the Faiss index and embeddings for future use
            faiss.write_index(self._faiss_index, self.index_file)
            np.save(self.embeddings_file, xb)

            # Save the document chunks for retrieval
            self._document_chunks = document_chunks
            self._embeddings = xb
        else:
            # Load the existing index and embeddings from files
            self._faiss_index = faiss.read_index(self.index_file)
            self._document_chunks = document_chunks  # Still need to pass the document chunks
            self._embeddings = np.load(self.embeddings_file)

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None, **kwargs) -> Prediction:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        embeddings = self._vectorizer(queries)
        emb_npa = np.array(embeddings)

        # For a single query, just look up the top k passages
        if len(queries) == 1:
            distance_list, index_list = self._faiss_index.search(emb_npa, k or self.k)
            passages = [(self._document_chunks[ind], ind, dist) for ind, dist in zip(index_list[0], distance_list[0])]
            return [dotdict({"long_text": passage[0], "index": passage[1], "distance": passage[2]}) for passage in passages]

        distance_list, index_list = self._faiss_index.search(emb_npa, (k or self.k) * 3, **kwargs)
        passage_scores = {}
        for emb in range(len(embeddings)):
            indices = index_list[emb]
            distances = distance_list[emb]
            for res in range((k or self.k) * 3):
                neighbor = indices[res]
                distance = distances[res]
                if neighbor in passage_scores:
                    passage_scores[neighbor].append(distance)
                else:
                    passage_scores[neighbor] = [distance]
        
        # Sorting the passages
        sorted_passages = sorted(passage_scores.items(), key=lambda x: (len(queries) - len(x[1]), sum(x[1])))[
            : k or self.k
        ]

        # Return results in docdict format with added metadata
        return [
            dotdict({"long_text": self._document_chunks[passage_index], "index": passage_index, "distance": sum(passage_scores[passage_index])})
            for passage_index, _ in sorted_passages
        ]