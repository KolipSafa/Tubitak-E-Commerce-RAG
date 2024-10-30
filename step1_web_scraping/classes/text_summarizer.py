from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json


prompt_template = """
You are an assistant that summarizes customer reviews for laptops.

Only include information that is explicitly stated in the input text.

Do not add any new information, assumptions, or details not present in the original text.

The summary should be written from the customer's perspective.

Return only the summarization text in a JSON object that has only a "review" (string) field that equals the summarization text.
"""

class TextSummarizer:
    def __init__(self, model_name="llama3.1:8b"):

        self.llm = ChatOllama(model=model_name,temperature=0.2, format="json")
        self.prompt = ChatPromptTemplate.from_messages([("system", prompt_template,), ("human", "{input}"),])
        self.chain = self.prompt | self.llm 
    
    def summarize(self, text):
        res = self.chain.invoke({ "input": text})
        return json.loads(res.content).get("review")

# sum = TextSummarizer()
# res = sum.summarize("If This isn't a backlight keyboard don't know what they're talking about.")

