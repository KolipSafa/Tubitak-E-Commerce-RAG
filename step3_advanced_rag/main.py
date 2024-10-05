import os
import faiss
import numpy as np
from utils import Utils
from dataset import Dataset
from colorama import Fore, Style
from tqdm import tqdm
from embeddings import Embeddings
from retriever import Retriever
from reranker import Reranker
from llm import LLM
import sys
import io
from dotenv import load_dotenv

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
    documents = dataset.get_dataset()

    # Generate embeddings for the documents using SentenceBERT and index them using FAISS
    print(f"{Fore.RED}2.) Generating Embedding Vectors using Sentence BERT and indexing using FAISS...{Style.RESET_ALL}")
    indices_path=os.getenv("LAPTOP_VECTOR_DB_PATH")
    sentence_bert = Embeddings("all-mpnet-base-v2")
    
    if not os.path.exists(indices_path):
        index = faiss.IndexFlatL2(768)
        batch_size = int(os.getenv("BATCH_SIZE"))
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding Documents", colour="green"):
            batch = documents[i:i+batch_size]
            embeds = sentence_bert.get_embedding(batch["text"])
            index.add(np.array(embeds))
            # Save the index
            faiss.write_index(index, indices_path)
    else:
        # Load the index
        index = faiss.read_index(indices_path)
    
    # Retrieve the top-k documents for a query using the FAISS index
    print(f"{Fore.RED}3.) Retrieve Top-K documents using FAISS...{Style.RESET_ALL}")
    query = "good battery life"
    retriever = Retriever()
    docs = retriever.search(documents=documents, embed_model=sentence_bert, index=index, query=query, top_k=20)
    print(docs)

    # Rerank the top-n documents using DistilBERT
    print(f"{Fore.RED}4.) Re-Ranking documents using distilBERT and retrieving Top-N documents...{Style.RESET_ALL}")
    reranker = Reranker("sentence-transformers/msmarco-distilbert-base-v3")
    reranked_docs = reranker.rerank(docs, query, top_n=5)
    context = "\n".join([doc[0] for doc in reranked_docs])
    
    # Generate response from OPENAI Model
    # print(f"{Fore.RED}5.) Generate reponse using LLM...{Style.RESET_ALL}")
    # llm = LLM(model=os.getenv("MODEL_NAME"), temperature=0)
    # llm_response = llm.generate(query=query, context=context)
    # print(f"Answer: {Fore.GREEN}{llm_response}{Style.RESET_ALL}")