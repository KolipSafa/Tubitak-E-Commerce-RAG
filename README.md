# Tubitak-E-Commerce-RAG

## To-do List

- [] Summarize and sanitize scraped reviews before chunking
- [] Add scraping for other devices
- [x] Update reranker for new faiss system
- [x] Change faiss to langchain faiss in step3

## Create env file

- Create .env file in root folder and populate with:

  ```shell
  LAPTOP_DB_PATH='data\db\laptops.db'
  LAPTOP_MARKDOWNS_PATH='data\markdown\laptops'
  LAPTOP_VECTOR_DB_PATH='data\vector_db\laptops.idx'
  LAPTOP_CHUNKED_DATASET_PATH='data/chunked/laptop_chunks.jsonl'

  OPENAI_API_KEY="your-openai-key"
  TOKENIZERS_PARALLELISM=true
  BATCH_SIZE=32
  MODEL_NAME="phi3.5:3.8b-mini-instruct-q6_K"
  ```

## step1_web_scraping Installation

- Download the dependencies

- To get laptop data run laptop_scraper.py

## step2_data_chunking Installation

- Download the dependencies

- To chunk data run data_chunker.py

## step3_advanced_rag Installation

To install and set up the Advanced RAG Pipeline, follow these steps:

- Install the required dependencies:

  ```shell
  pip install -r requirements.txt
  ```

- Run the pipeline:

  ```shell
  python main.py
  ```
