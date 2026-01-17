# https://stackoverflow.com/questions/77284901/upload-an-image-to-chat-gpt-using-the-api

import base64
import os
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image


load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

# Base URL override (if unset/blank, default OpenAI endpoint is used)
base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"

# Models
MODEL_NANO = os.getenv("OPENAI_MODEL_LIGHT", "gpt-5-nano")

# Function to encode the image
def encode_image(image_path):
    img = Image.open(image_path)
    newsize = (256, 256)
    img = img.resize(newsize)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode('utf-8')


# Path to your image
image_path = "views/Bacteriophage/parts/change.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": MODEL_NANO,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the shape of this dinosaur-headed robot in detail"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
}

response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)

print(response.json())
