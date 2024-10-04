from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

class LLM:
    """LLM Class for RAG Ppeline thats applied after Re-ranking documents
    """
    def __init__(self, model: str, temperature: int) -> None:
        self.model = model
        self.temperature = temperature
        
    def generate(self, query: str, context: str) -> str:
        """Generate a response from an LLM using the given context
        """
        summary = """
        You're an Technology Consultant to answer questions using the given context.

        Context: {context}
        
        I want you to find and return most similar ids of laptops that I have described below. 
        
        Laptop Description: {query}
        """
        llm = ChatOllama(temperature=self.temperature, model=self.model)
        prompt = ChatPromptTemplate(input_variables=["context"], template=summary)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(input={"context": context, "query": query})
        return result
