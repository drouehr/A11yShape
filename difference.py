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

views_folder = 'temp'


# Function to encode the image
def encode_image(image_path):
    img = Image.open(image_path)
    newsize = (256, 256)
    img = img.resize(newsize)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode('utf-8')


def get_difference(code, prev):
    
    cl = code.split('module')
    pl = prev.split('module')

    diff = []
    
    if len(cl) == len(pl):
        for i in range(0, len(cl)):
            if cl[i] != pl[i]:
                #print('module'+pl[i])
                diff.append({'bef': 'module'+pl[i], 'aft': 'module'+cl[i]})
    if len(diff) == 0:
        return 'no difference'
    

    file = 'before.scad'
    f = open(join(views_folder, file), "w")
    f.write(prev)
    f.close()
    fp = join(views_folder, file)
    
    outfile1 = fp.replace('.scad', '.png')
    cmd = 'openscad -o '+outfile1+' -q --camera 0,0,0,0,0,0 --viewall --autocenter --imgsize=2048,2048 '+fp
    os.system(cmd)
    
    file = 'after.scad'
    f = open(join(views_folder, file), "w")
    f.write(code)
    f.close()
    fp = join(views_folder, file)
    
    outfile2 = fp.replace('.scad', '.png')
    cmd = 'openscad -o '+outfile2+' -q --camera 0,0,0,0,0,0 --viewall --autocenter --imgsize=2048,2048 '+fp
    os.system(cmd)
    
    

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    base64_image = encode_image(outfile1)
    base64_image2 = encode_image(outfile2)

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

                """+diff[0]['bef']+"""

                After:

                """+diff[0]['aft']+"""
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
    
code = """
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

#print(get_difference(code, prev))



    