import json
import os
import faiss
import numpy as np
import sys
import io
from uuid import uuid4
from utils import Utils
from dataset import Dataset
from colorama import Fore, Style
from tqdm import tqdm
from embeddings import Embeddings
from retriever import Retriever
from reranker import Reranker
from llm import LLM
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

if __name__ == '__main__':
    print(f"{Fore.YELLOW}=====Advanced RAG Pipeline====={Style.RESET_ALL}")
    # Load Utils
    utils = Utils()
    utils.check_dir(".indices")
    load_dotenv()
    
    # Load the dataset (train split) that's already chunked
    print(f"{Fore.RED}1.) Loading and chunking dataset...{Style.RESET_ALL}")
    print(os.getenv("LATOP_CHUNKED_DATASET_PATH"))
    dataset = Dataset(os.getenv("LATOP_CHUNKED_DATASET_PATH"), "train")
    documents = dataset.get_dataset_as_documents()

    # Generate embeddings for the documents using SentenceBERT and index them using FAISS
    print(f"{Fore.RED}2.) Generating Embedding Vectors using Sentence BERT and indexing using FAISS...{Style.RESET_ALL}")
    indices_path=os.getenv("LAPTOP_VECTOR_DB_PATH")
    sentence_bert = Embeddings("all-mpnet-base-v2")
    
    if not os.path.exists(indices_path):
        index = faiss.IndexFlatL2(768)

        vector_store = FAISS(
            embedding_function=sentence_bert.get_model().encode,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        batch_size = int(os.getenv("BATCH_SIZE"))
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding Documents", colour="green"):
            batch = documents[i:i+batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch))]
            vector_store.add_documents(documents=batch, ids=uuids)

        # Save the index
        vector_store.save_local(indices_path)
    else:
        vector_store = FAISS.load_local(indices_path, embeddings=sentence_bert.get_model().encode,allow_dangerous_deserialization=True)
    
    # Retrieve the top-k documents for a query using the FAISS index
    print(f"{Fore.RED}3.) Retrieve Top-K documents using FAISS...{Style.RESET_ALL}")
    query = "i need a laptop for playing gta v"
    retriever = Retriever()
    docs_with_scores = retriever.search(vector_store=vector_store, query=query, top_k=10)
    # print(docs)
    utils.save_docs_with_scores(docs_with_scores, 'data/retrieved_documents.json')

    #Rerank the top-n documents using DistilBERT
    print(f"{Fore.RED}4.) Re-Ranking documents using distilBERT and retrieving Top-N documents...{Style.RESET_ALL}")
    docs = [doc for doc, score in docs_with_scores]
    reranker = Reranker("sentence-transformers/msmarco-distilbert-base-v3")
    reranked_docs_with_scores = reranker.rerank(docs, query, top_n=5)
    utils.save_docs_with_scores(reranked_docs_with_scores, 'data/reranked_documents.json')
    # context = "\n".join([doc[0] for doc in reranked_docs])
    # context = "\n".join([doc.page_content for doc, score in reranked_docs_with_scores])
    
    # Generate response from OPENAI Model
    # print(f"{Fore.RED}5.) Generate reponse using LLM...{Style.RESET_ALL}")
    # llm = LLM(model=os.getenv("MODEL_NAME"), temperature=0)
    # llm_response = llm.generate(query=query, context=context)
    # print(f"Answer: {Fore.GREEN}{llm_response}{Style.RESET_ALL}")