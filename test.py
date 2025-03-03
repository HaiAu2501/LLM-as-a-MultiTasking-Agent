import os
import hydra
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from utils.client import init_client

load_dotenv()

messages = [
    {"role": "system", "content": "You are a chatbot."},
    {"role": "user", "content": "Write a function that takes a list of numbers and returns the sum."}
]

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    client = init_client(cfg)
    response = client.get_response(messages)
    print(response)

if __name__ == "__main__":
    main()
    