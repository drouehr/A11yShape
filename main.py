# https://stackoverflow.com/questions/77284901/upload-an-image-to-chat-gpt-using-the-api

from difference import get_difference

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
                #"text": "Given an object viewed from different angles, describe the shape of this object in detail such that a blind user could understand it"
                "text": "Given the same 3D model viewed from different angles, describe the shape such that a blind user could understand it"
                #"text": "describe how the changes in the openscad code would change the resulting 3d model such that a blind person could understand it"
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


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
    

def run(req, model):
    #print('running')
    code = req['code']
    if 'prev' in req:
        d = get_difference(code, req['prev'])
        response = {'description': d}
        return response
    
    
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
    
    #angles = [[-1,0,0], [0,-1,0], [0,0,-1], [0,0,1], [1,0,0], [0,1,0]]
    #angles = [[-1,1,1], [0,-1,1], [1,-1,1], [-1,-1,1], [-1,1,-1], [-1,1,0]]
    angles = [[-1,-1,0], [1,-1,0]]
    imgs = []

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
        imgs.append(outfile)
        #if os.path.exists(outfile):
        #    continue

        cmd = 'openscad -o '+outfile+' -q --camera '+str(x)+','+str(y)+','+str(z)+',0,0,0 --viewall --autocenter --imgsize=2048,2048 '+fp
        #print(cmd)
        os.system(cmd)
        
    
    img = Image.open(imgs[0])
    for i in imgs[1:]:
        img2 = Image.open(i)
        img = get_concat_h(img, img2)
    
    image_path = join(outpath, 'combined.png')
    img.save(image_path)
    
    base64_image = encode_image(image_path)
    d = desc_gpt4(base64_image)
    #print(d)

    """

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
    """
    response = {'description': d}
    return response




prev = """
    // Bacteriophage by Erik Wrenholt 2017-02-12
// License: Creative Commons - Attribution

$fn = 12;

leg_count = 6;
leg_width = 1.75;

printable_phage();

module printable_phage() {
    // chop off the bottom of the legs so they are flat on the bottom.
    bacteriophage();
}

module bacteriophage() {
    body();
    for(i=[0:leg_count]) {
        rotate((360 / leg_count) * i, [0,0,1])
            leg();
    }
 }

module body() {
    
    // Icosahedral head
    translate([0,0,30]) 
    scale([1,1,1.3])
        rotate(30, [0,1,0]) 
                icosahedron(5);
    
    // Base-Plate
    translate([0,0,1.5])
        scale([1,1,0.4])
            rotate(30, [0,0,1])
                sphere(6, $fn=12);

    // Helical Sheath
    for(i=[2:10]) {
        translate([0,0,i*2])
            scale([1,1,0.5])
                sphere(4);
    }

}

module leg() {
    union() {
        hull() {
            translate([2,0,0]) sphere(leg_width);
            translate([15,0,12]) sphere(leg_width);
        }
        hull() {
            translate([15,0,12]) sphere(leg_width);
            translate([25,0,-2]) sphere(leg_width);
        }
    }
}


// https://www.thingiverse.com/thing:1343285

/*****************************************************************
* Icosahedron   By Adam Anderson
* 
* This module allows for the creation of an icosahedron scaled up 
* by a factor determined by the module parameter. 
* The coordinates for the vertices were taken from
* http://www.sacred-geometry.es/?q=en/content/phi-sacred-solids
*************************************************************/

module icosahedron(a = 2) {
    phi = a * ((1 + sqrt(5)) / 2);
    polyhedron(
        points = [
            [0,-phi, a], [0, phi, a], [0, phi, -a], [0, -phi, -a], [a, 0, phi], [-a, 0, phi], [-a, 0, -phi], 
            [a, 0, -phi], [phi, a, 0], [-phi, a, 0], [-phi, -a, 0], [phi, -a, 0]    
        ],
        faces = [
            [0,5,4], [0,4,11], [11,4,8], [11,8,7], [4,5,1], [4,1,8], [8,1,2], [8,2,7], [1,5,9], [1,9,2], [2,9,6], [2,6,7], [9,5,10], [9,10,6], [6,10,3], [6,3,7], [10,5,0], [10,0,3], [3,0,11], [3,11,7]
        ]
    
    );
}
    """


testing = """
    // Bacteriophage by Erik Wrenholt 2017-02-12
// License: Creative Commons - Attribution

$fn = 12;

leg_count = 6;
leg_width = 1.75;

printable_phage();

module printable_phage() {
    // chop off the bottom of the legs so they are flat on the bottom.
    difference() {
        bacteriophage();
        translate([0,0,-5]) cube([100,100,10], center=true);
    }
}

module bacteriophage() {
    body();
    for(i=[0:leg_count]) {
        rotate((360 / leg_count) * i, [0,0,1])
            leg();
    }
 }

module body() {
    
    // Icosahedral head
    translate([0,0,30]) 
    scale([1,1,1.3])
        rotate(30, [0,1,0]) 
                icosahedron(5);
    
    // Base-Plate
    translate([0,0,1.5])
        scale([1,1,0.4])
            rotate(30, [0,0,1])
                sphere(6, $fn=12);

    // Helical Sheath
    for(i=[2:10]) {
        translate([0,0,i*2])
            scale([1,1,0.5])
                sphere(4);
    }

}

module leg() {
    union() {
        hull() {
            translate([2,0,0]) sphere(leg_width);
            translate([15,0,12]) sphere(leg_width);
        }
        hull() {
            translate([15,0,12]) sphere(leg_width);
            translate([25,0,-2]) sphere(leg_width);
        }
    }
}


// https://www.thingiverse.com/thing:1343285

/*****************************************************************
* Icosahedron   By Adam Anderson
* 
* This module allows for the creation of an icosahedron scaled up 
* by a factor determined by the module parameter. 
* The coordinates for the vertices were taken from
* http://www.sacred-geometry.es/?q=en/content/phi-sacred-solids
*************************************************************/

module icosahedron(a = 2) {
    phi = a * ((1 + sqrt(5)) / 2);
    polyhedron(
        points = [
            [0,-phi, a], [0, phi, a], [0, phi, -a], [0, -phi, -a], [a, 0, phi], [-a, 0, phi], [-a, 0, -phi], 
            [a, 0, -phi], [phi, a, 0], [-phi, a, 0], [-phi, -a, 0], [phi, -a, 0]    
        ],
        faces = [
            [0,5,4], [0,4,11], [11,4,8], [11,8,7], [4,5,1], [4,1,8], [8,1,2], [8,2,7], [1,5,9], [1,9,2], [2,9,6], [2,6,7], [9,5,10], [9,10,6], [6,10,3], [6,3,7], [10,5,0], [10,0,3], [3,0,11], [3,11,7]
        ]
    
    );
}
    """

req = {'code': testing, 'prev': prev}

req = {"code":"$fn=32; sphere();"}
print(run(req, 'gpt4'))