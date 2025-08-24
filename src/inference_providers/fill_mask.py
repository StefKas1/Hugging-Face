import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.fill_mask(
    "The answer to the universe is [MASK].",
    model="google-bert/bert-base-uncased",
)

print(result)
