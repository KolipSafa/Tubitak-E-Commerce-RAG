# Tubitak-E-Commerce-RAG

## step1_web_scraping Installation

- Download the dependencies
- Create .env file in root folder and populate with:
  ```
  LAPTOP_DB_PATH='data\db\laptops.db'
  LAPTOP_MARKDOWNS_PATH='data\markdown\laptops'
  LAPTOP_VECTOR_DB_PATH='data\vector_db\laptops.index'
  ```
- To get laptop data run laptop_scraper.py

## step2_data_chunking Installation

- Download the dependencies

- To get laptop data run data_chunker.py

## step3_advanced_rag Installation

To install and set up the Advanced RAG Pipeline, follow these steps:

- Install the required dependencies:

  ```shell
  pip install -r requirements.txt
  ```

- Configure the pipeline:

  - Update the environment file `.env` with the following settings:

  ```shell
  OPENAI_API_KEY="your-openai-key"
  TOKENIZERS_PARALLELISM=true
  BATCH_SIZE=32
  MODEL_NAME="model-name"
  ```

- Run the pipeline:

  ```shell
  python main.py
  ```
