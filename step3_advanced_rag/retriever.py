from embeddings import Embeddings

class Retriever:
    @staticmethod
    def search(vector_store, query, top_k) -> list:
        return vector_store.similarity_search_with_relevance_scores(query=query, k=top_k)
