import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

output = client.image_segmentation(
    "cats.jpg", model="jonathandinu/face-parsing"
)  # Input not provided in repo
