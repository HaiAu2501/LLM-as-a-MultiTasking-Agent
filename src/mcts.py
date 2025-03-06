import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from utils.client import init_client

load_dotenv()

messages = [
    {"role": "system", "content": "You are a chatbot."},
    {"role": "user", "content": "Write a function that takes a list of numbers and returns the sum."}
]

@hydra.main(config_path="../cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    client = init_client(cfg)
    code, _ = client.get_code(messages)

if __name__ == "__main__":
    main()
    