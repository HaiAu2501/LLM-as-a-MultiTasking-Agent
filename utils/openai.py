import os
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from utils.format.code import Code
from utils.format.decision import OptimizationDecision

class LLMClient:
    """
    Unified LLM client using OpenAI API for both GPT and Gemini models.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1,
    ) -> None:
        """
        Initialize the LLM client.
        
        Args:
            model: The model name (either gpt-* or gemini-*)
            temperature: The temperature parameter for generation
            api_key: The API key for OpenAI
        """
        self.model = model
        self.temperature = temperature

        base_url = None
        if model.startswith("gpt"):
            api_key = os.getenv("GPT_API_KEY")
        elif model.startswith("gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = os.getenv("GEMINI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def get_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """
        Generate responses using the configured model via OpenAI API.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            temperature: Override the default temperature if provided
            
        Returns:
            List of response dictionaries
        """
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            stream=False,
        ).choices[0].message.content
        return response

    
    def get_multiple_responses(self, messages_list: List[List[Dict[str, str]]], temperature: Optional[float] = None) -> List[str]:
        """
        Generate responses for multiple message lists.
        
        Args:
            messages_list: List of message lists, each containing message dictionaries
            temperature: Override the default temperature if provided
            
        Returns:
            List of response content strings
        """

        responses = []
        for messages in messages_list:
            response = self.get_response(messages, temperature)
            responses.append(response)
        return responses

    def get_code(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Tuple[str, str]:
        """
        Generate code completions using the configured model via OpenAI API.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            temperature: Override the default temperature if provided
            
        Returns:
            Code snippet and explanation
        """
        temp = temperature if temperature is not None else self.temperature

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            response_format=Code,
        ).choices[0].message.parsed

        return response.code, response.explain
        
    def get_optimization_decision(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Tuple[str, str, str]:
        """
        Get the decision on which function to optimize next from the LLM.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            temperature: Override the default temperature if provided
            
        Returns:
            Tuple containing the function_id, rationale, and suggestions
        """
        temp = temperature if temperature is not None else self.temperature
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
            response_format=OptimizationDecision,
        ).choices[0].message.parsed
        
        return response.function_id, response.rationale, response.suggestions