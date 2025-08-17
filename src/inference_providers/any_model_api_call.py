# 1. Search a model: https://huggingface.co/models
# 2. Click: “View Code Snippets”
# 3. (Adjust and) use code

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)

image.save("Astronaut riding a horse.png")
