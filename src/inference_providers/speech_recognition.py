import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

output = client.automatic_speech_recognition(
    "sample1.flac",
    model="openai/whisper-large-v3",  # Input not provided in repo
)
print(output)
