from omegaconf import DictConfig

def init_client(cfg: DictConfig):
    global client
    from .openai import LLMClient
    llm_config = cfg.llm
    client = LLMClient(model=llm_config.model, temperature=llm_config.temperature)
    return client