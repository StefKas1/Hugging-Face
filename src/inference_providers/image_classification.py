import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

output = client.image_classification(
    "cats.jpg", model="Falconsai/nsfw_image_detection"
)  # Input not provided in repo
print(output)
