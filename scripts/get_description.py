# https://stackoverflow.com/questions/77284901/upload-an-image-to-chat-gpt-using-the-api

import base64
import requests
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image


load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_KEY")

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
    "model": "gpt-4-vision-preview",
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

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
