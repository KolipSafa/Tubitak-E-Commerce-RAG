from sentence_transformers import SentenceTransformer, util


class Reranker:
    """
    Reranks the top-k documents retrieved by the retriever using a SentenceTransformer model.
    """
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(self, documents, query, top_n=10) -> list:
        model = SentenceTransformer(self.model_name)

        query_embedding = model.encode(query, convert_to_tensor=True)
        document_texts = [doc.page_content for doc in documents]
        document_embeddings = model.encode(document_texts, convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
        doc_scores = list(zip(documents, similarities))
    
        reranked_documents = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        reranked_documents = reranked_documents[:top_n]
    
        return reranked_documents