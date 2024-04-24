import os
import time

from dotenv import load_dotenv
from openai import OpenAI


# 주어진 문자열을 임베딩하는 함수
def embed(text: str, model: str, wait_time: float = 0.1) -> list:
    response = client.embeddings.create(input=text, model=model)
    time.sleep(wait_time)
    embedding = response.data[0].embedding
    return embedding


# 주어진 프롬프트에 따라 답변을 생성하는 함수
def generate(model: str, messages=list) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    response = response.choices[0].message.content
    return response


# API key 설정
load_dotenv()
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# 사용할 LLM 설정
client = OpenAI()
