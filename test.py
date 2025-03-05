import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from utils.client import init_client

load_dotenv()

messages = [
    {"role": "system", "content": "You are a chatbot."},
    {"role": "user", "content": "Write a function that takes a list of numbers and returns the sum."}
]

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    cfg.llm.model = "gemini-2.0-flash"
    client = init_client(cfg)

    code, _ = client.get_code(messages)
    print(code)

if __name__ == "__main__":
    main()
    