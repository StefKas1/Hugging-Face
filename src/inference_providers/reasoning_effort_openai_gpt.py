from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-120b:fireworks-ai",
    instructions="You are a helpful assistant.",
    input="Say hello to the world.",
    reasoning={
        "effort": "low",  # Controls model's thinking time
    },
)

for index, item in enumerate(response.output):
    print(f"Output #{index}: {item.type}", item.content)
