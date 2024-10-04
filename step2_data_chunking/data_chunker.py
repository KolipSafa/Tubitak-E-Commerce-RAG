import os
import sys
import json
from dotenv import load_dotenv
from typing import Sequence
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode, Document

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L12-v2')

# Initialize the node parser with parameters
node_parser = SemanticSplitterNodeParser(buffer_size=50, breakpoint_percentile_threshold=80, embed_model=embed_model)

# Initialize an empty list to hold all the documents
documents: Sequence[Document] = []

# Define the base path to the laptop directories
base_path = str(os.getenv('LAPTOP_MARKDOWNS_PATH'))

# Iterate over each laptop directory
for laptop_id in os.listdir(base_path):
    laptop_path = os.path.join(base_path, laptop_id)
    if os.path.isdir(laptop_path):
        # Use SimpleDirectoryReader to read all markdown files in the laptop directory
        reader = SimpleDirectoryReader(input_dir=laptop_path, file_extractor={'md': 'markdown'})
        laptop_documents: list[Document]  = reader.load_data()

        for doc in laptop_documents:
            doc.metadata['laptop_id'] = laptop_id

        documents.extend(laptop_documents)

# Parse the documents into nodes (chunks)
nodes: list[BaseNode] = []

for doc_id, doc in enumerate(documents):
    parsed_nodes = node_parser.get_nodes_from_documents([doc])
    for node in parsed_nodes:
        node.metadata['document_id'] = doc_id  # Add document ID for tracking
        nodes.append(node)

# Prepare the data for JSONL format
jsonl_data = []

for i, node in enumerate(nodes):
    chunk_data = {
        'chunk_id': i,
        'document_id': node.metadata.get('document_id'),
        'laptop_id': node.metadata.get('laptop_id'),
        'chunk': node.get_content()
    }
    jsonl_data.append(chunk_data)

# Save the chunked data in JSONL format
jsonl_path = 'data/chunked/laptop_chunks.jsonl'
with open(jsonl_path, 'w', encoding='utf-8') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False,indent=4) + '\n')

print("Chunked data saved to JSONL format.")
