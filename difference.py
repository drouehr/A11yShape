import base64
import requests
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image
import json

from os import listdir
from os.path import isfile, join

#import replicate

load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_KEY")

rep_key = os.getenv("REPLICATE_API_TOKEN")


# Function to encode the image
def encode_image(image_path):
    img = Image.open(image_path)
    newsize = (256, 256)
    img = img.resize(newsize)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode('utf-8')


def desc_gpt4():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    base64_image = encode_image('temp/d1.png')
    base64_image2 = encode_image('temp/d2.png')

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                #"text": "Given an object viewed from different angles, describe the shape of this object in detail such that a blind user could understand it"
                #"text": "Given the same 3D model viewed from different angles, describe the shape such that a blind user could understand it"
                "text": "describe how the changes in the openscad code would change the resulting 3d model such that a blind person could understand it"
              },
              {
                "type": "text",
                "text": """
                Before:

                bacteriophage();

                After:

                difference() {
                    bacteriophage();
                    translate([0,0,-5]) cube([100,100,10], center=true);
                }
                """
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image2}"
                }
              }
            ]
          }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    #print(response)
    return response['choices'][0]['message']['content']
    



print(desc_gpt4())



    