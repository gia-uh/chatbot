from pydantic import BaseModel
import yaml
import dotenv
import os

dotenv.load_dotenv()


class TelegramConfig(BaseModel):
    token: str


class LLMConfig(BaseModel):
    model: str
    api_key: str
    base_url: str
    
class EmbeddingConfig(BaseModel):
    model: str
    api_key: str
    base_url: str


class PromptsConfig(BaseModel):
    system: str


class ConversationConfig(BaseModel):
    history: int = 5


class Config(BaseModel):
    telegram: TelegramConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    conversation: ConversationConfig
    prompts: PromptsConfig
    db: str = "bot.db"
    start: str


def patch(data: dict):
    for k, v in data.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            data[k] = os.getenv(v[2:-1])
        elif isinstance(v, dict):
            patch(v)


def load(path: str = "config.yaml") -> Config:
    with open(path) as fp:
        data = yaml.safe_load(fp)

        patch(data)

        return Config(**data)
