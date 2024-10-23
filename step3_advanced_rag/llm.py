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
        You are a Technology Consultant tasked with finding the most suitable laptops based on the user's request, using the context provided within the <given_context> tags.

        <instructions>
        1.Understand the User's Request:
        - Carefully read the user's request to grasp their specific needs and preferences.

        2.Assess Each Laptop:
        - For each laptop in the given context, evaluate how closely it matches the user's requirements.
        - Assign a score from 1 to 5, where:
            - 1 = Very low match
            - 2 = Low match
            - 3 = Average match
            - 4 = High match
            - 5 = Excellent match

        3.Write a Summary:
        - In the "summary" field, provide a brief, colloquial explanation of how well the laptop meets the user's needs.
        - Mention relevant features and any missing aspects.
        - Do not mention or refer to the "laptop_id" or any other identifiers.

        4.Output Format:
        - Output a JSON array of objects, one for each laptop, the schema defined in the `<output_json_schema>` tags.
        - Do not include any additional text or explanations; only return the JSON object.
        </instructions>

        <output_json_schema>
        - "laptop_id" (string): The laptop's identifier.
        - "score" (integer): A number between 1 and 5 indicating how closely the laptop matches the user's request.
        - "summary" (string): A brief, colloquial summary explaining how well the laptop matches the user's needs and any missing features.
        </output_json_schema>

        The following <given_context> tags include the given context that have each laptop's id, technical specifications and review.
        <given_context>
            {context}
        </given_context>
        """

        user_prompt = """
        User's Request:
        {query}
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
