import json
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from utils import Utils

load_dotenv(find_dotenv())

class LLM:
    """LLM Class for RAG Pipeline thats applied after Re-ranking documents
    """
    def __init__(self, model: str, temperature: int) -> None:
        self.model = model
        self.temperature = temperature
        
    def generate(self, query: str, context: str) -> str:
        """Generate a response from an LLM using the given context"""
        system_prompt = """
        You are a Technology Consultant tasked with finding the most suitable laptops based on the user's description.

        You must assess the descriptions to determine how closely they match the user's query, providing a score from 1 to 5 where:
        - 1 = Very low match
        - 2 = Low match
        - 3 = Average match
        - 4 = High match
        - 5 = Excellent match


        Your task is to strictly output the response in valid JSON format, with the following structure for each laptop:

        - "laptop_id": The laptop's identifier (string)
        - "score": A number between 1 and 5 indicating similarity to the user's query (integer)
        - "summary": A brief, colloquial summary explaining how well the laptop matches the user's needs and any missing features (string)

        In the summary field, **do not mention or refer to the laptop_id**. Focus only on the features and suitability of the laptop for the user's needs without including any identifiers.

        The following context includes each laptop's id, its description and its similarity score to the user's query.

        Context:
            {context}
        """

        user_prompt = """
        User Description:
        {query}

        Output the JSON array of objects for each laptop. Do not include any other text or explanations, just return the JSON object.
        """
        
        llm = ChatOllama(temperature=self.temperature, model=self.model)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt.strip()),
            HumanMessagePromptTemplate.from_template(user_prompt.strip()),
        ])

        messages = prompt.format_messages(context=context, query=query)

        response = llm.invoke(messages)
        
        # Parse the response
        parsed_response = StrOutputParser().parse(response.content)

        full_prompt = "".join([message.content for message in messages])

        Utils.save_llm_result(full_prompt,response.content,'data/llm_result.json')
        
        return parsed_response
