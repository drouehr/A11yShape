# https://stackoverflow.com/questions/77284901/upload-an-image-to-chat-gpt-using-the-api

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

models_folder = 'models'
views_folder = 'temp'

file = 'Bacteriophage.scad'


# Function to encode the image
def encode_image(image_path):
    img = Image.open(image_path)
    newsize = (256, 256)
    img = img.resize(newsize)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode('utf-8')


def desc_gpt4(base64_image):
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
                "text": "Describe the shape of this object in detail such that a blind user could understand it"
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
    response = response.json()
    #print(response)
    return response['choices'][0]['message']['content']

"""
def desc_blip(base64_image):
    iterator = replicate.run(
        "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe",
        input={
            "img": f"data:image/jpeg;base64,{base64_image}",
            "seed": -1,
            "debug": False,
            "top_k": 0,
            "top_p": 1,
            "prompt": "Describe this object's appearance in detail",
            "max_length": 512,
            "temperature": 0.75,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "repetition_penalty": 1,
            "no_repeat_ngram_size": 0
        }
    )
    output = ""
    for text in iterator:
      output = output+text
    return output
"""


def run(req, model):
    code = req['code']
    file = 'model.scad'
    f = open(join(views_folder, file), "w")
    f.write(code)
    f.close()

    #print(file)

    #print(model)
    fp = join(views_folder, file)
    outpath = join(views_folder, file.rsplit('.', 1)[0])

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    desc = []
    prompt = """
    Given a set of descriptions about the same 3D object, distill these descriptions into one detailed description such that a blind user could understand it

    """
    
    angles = [[-1,0,0], [0,-1,0], [0,0,-1], [0,0,1], [1,0,0], [0,1,0]]

    """
    for x in range(-1,2):
        for y in range(-1,2):
            for z in range(-1,2):
                if x==0 and y==0 and z==0:
                    continue
    """
    for ang in angles:
        x = ang[0]
        y = ang[1]
        z = ang[2]
        
        outfile = join(outpath, str(x+1)+str(y+1)+str(z+1)+'.png')
        #if os.path.exists(outfile):
        #    continue

        cmd = 'openscad -o '+outfile+' -q --camera '+str(x)+','+str(y)+','+str(z)+',0,0,0 --viewall --autocenter --imgsize=2048,2048 '+fp
        #print(cmd)
        os.system(cmd)



        # Path to your image
        image_path = outfile #"temp/Bacteriophage/"+str(x+1)+str(y+1)+str(z+1)+'.png'
        #return cmd

        # Getting the base64 string
        base64_image = encode_image(image_path)


        if model == 'gpt4':
            try:
                d = desc_gpt4(base64_image)
            except Exception as e:
                d = desc_gpt4(base64_image)
        #else:
            #d = desc_blip(base64_image)
        #print(d)
        desc.append(d)
        prompt = prompt + d+'\n\n'


    #print(desc)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": prompt
              },
            ]
          }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    d = response['choices'][0]['message']['content']
    #print(d)
    
    response = {'description': d}
    return response


testing = """
    $fn=32;
    cube();
    sphere();
    """

#run(testing, 'gpt4')