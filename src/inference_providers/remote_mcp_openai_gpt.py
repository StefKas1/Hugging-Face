import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-120b:fireworks-ai",
    input="What transport protocols are supported in the 2025-03-26 version of the MCP spec?",
    tools=[
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": "never",
        },
    ],
)

print(response)
