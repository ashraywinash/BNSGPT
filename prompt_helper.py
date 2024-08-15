from typing import Any, List
from langchain_core.outputs import LLMResult
import requests

from langchain_core.prompts import PromptTemplate

class BNSPrompter():
    def __init__(self) -> None:
        self.prompt_template: str = """
        You are a legal expert's assistant specializing in criminal law. Based on the following details, provide a concise summary:
        - **Events**: {events}
        - **Applicable BNS Law Sections**: {sections}
        Please summarize the relevant law sections. Ensure that your response is straightforward and easy to understand and under 200 words, highlighting the sections,key points and reasoning behind each applicable law section.
        """


    def get_prompt(self, events : str, sections:str):
        prompt=PromptTemplate.from_template(template=self.prompt_template)
        prompt_txt = prompt.format(events=events, sections=sections)
        return prompt_txt




class BNS_LLAMA_CustomModel():

    def __init__(self, model) -> None:
        self.model = model

    def _llm_type(self) -> str:
        return "custom"

    def _generate(self, prompts: List[str]) -> LLMResult:
        prompt = " ".join(prompts)

        # Define the API endpoint and payload
        url = 'http://localhost:11434/api/generate'
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # Send POST request
        response = requests.post(url, json=payload)

        # Check for request errors
        assert response.status_code == 200
        response_json = response.json()

        return response_json

    def call(self, prompt: str) -> str:
        # Generate response
        result = self._generate([prompt])

        # Extract and return the text from LLMResult
        return result

# Instantiate and use the custom model
# cm = CustomModel(model="llama2")
# result = cm.call(formatted_prompt)
# print(result["response"])
