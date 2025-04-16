"""
This code defines a Flask application that serves as an API for generating STL files, saving and loading OpenSCAD code files, and performing model description and code matching tasks using OpenAI's GPT-4 models.

The API endpoints include:
- /generate-stl: Accepts a POST request with OpenSCAD code and generates an STL file.
- /save-code: Accepts a POST request with OpenSCAD code and saves it as a file.
- /get-files: Returns a list of saved code files.
- /load-code/<file_name>: Returns the content of a specific code file.
- /api/describe: Accepts a POST request with text, code, and an image, and uses GPT-4 to generate a description of the 3D model.
- /api/match: Accepts a POST request with text, code, and an image, and uses GPT-4 to match the different parts of the 3D model to the corresponding code.
- /api/analysis: Accepts a POST request with text and code, and uses GPT-4 to analyze the OpenSCAD code.
- /api/improve: Accepts a POST request with text and code, and uses GPT-4 to provide suggestions for improving the code.
The code also includes configuration settings for the GPT-4 models, agent definitions for model descriptor, code interpreter, and user proxy, and a Flask route for serving the index.html file.

Note: The code includes sensitive information such as API keys and authorization headers. Make sure to handle this information securely in a production environment.
"""
from flask import (
    Flask,
    request,
    send_from_directory,
    jsonify,
    Response,
    stream_with_context,
)
from flask_cors import CORS, cross_origin
import base64
import requests
import logging
import os
from os.path import isfile, join
import subprocess
from openai import OpenAI 
import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
import requests
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from PIL import Image
import io
import json
import time


logging.basicConfig(level=logging.INFO)

api_key = "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"
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", 'Insert you api key here'))
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))

"""AutoGen Config"""
config_llm_4v = [{"model": "gpt-4o", "api_key": api_key}]

config_llm_4 = [{"model": "gpt-4o", "api_key": api_key}]

config_list_4v = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_llm_4v,
    "temperature": 0,
    "max_tokens": 2000,
}

config_list_4 = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_llm_4,
    "temperature": 0,
}

model_descriptor = MultimodalConversableAgent(
    name="3D_model_descriptor",
    max_consecutive_auto_reply=10,
    llm_config=config_list_4v,
    system_message="""
As a good 3D model descriptor, you will receive images from the OpenSCAD 3D model and generate a detailed description of the 3D model, describing what the 3D model is and what parts it consists of. After that, you will work with the code interpreter to match the different parts of the model to the code that generates this corresponding part.
Use the following format for output:
***Report Begins***
##Description of the model##
[Insert the description of the model here, highlighting key elements.]

##Summary of the model##
[Insert the summary of the model here, contains all the components.]
***Report Ends***
""",
)

code_interpreter = autogen.AssistantAgent(
    name="code_interpreter",
    llm_config=config_list_4,
    system_message="""
As an expert in OpenSCAD code interpretation, you will receive a set of OpenSCAD code. For a given piece of code, you will work with the 3D model descriptor to connect the different parts of the 3d model and their corresponding code.
Use the following format for output:
***Report Begins***
##Codes##
"Code1", [The corresponding part in the model], 
[content of Code1]
"Code2", [The corresponding part in the model], 
[content of Code1]
...
***Report Ends***
""",
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",  # Try between ALWAYS or NEVER
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

groupchat = autogen.GroupChat(
    agents=[model_descriptor, code_interpreter,
            user_proxy], messages=[], max_round=5
)
manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=config_list_4)


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
PORT = int(os.environ.get("PORT", 3000))




@app.route('/')
def hello_world():
    return 'Hello from Flask6!'

@app.route('/code2fab', methods=['POST'])
def code2fab():
    req = request.json
    #return jsonify(req['code'])
    return jsonify(request.json)



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode("utf-8")

def upload_image(image_path):
    headers = {'Authorization': 'ICcEBQDFvmdJGPfwpGMSYxgkSEYHnVyw'}
    url = "https://sm.ms/api/v2/upload"
    files = {"smfile": open(image_path, "rb")}
    response = requests.post(url, files=files, headers = headers)
    if response.status_code == 200:
        data = response.json()
        if data["code"] == "success":
            return data["data"]["url"]
    return None

current_process = None

def find_openscad_errors(code):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        timeout=10,
        #stream=True,
        messages=[
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": 'Describe all of the errors the OpenSCAD code. Each error should be outputted as a new line. Output only the errors and nothing else'},
                    {"type": "text", "text": addLineNum(code)},
                ],
            }
        ],
    )
    return response.choices[0].message.content.replace('`', '')


def gen_image(views, code, output_dir):
    code = code.encode('ascii',errors='ignore').decode().lower()
    
    file = 'model.scad'
    f = open(join(output_dir, file), "w")
    f.write(code)
    f.close()
    
    encoded_imgs = []
    encoded_imgs_sm = []
    
    processes = []
    for index in views:
        view = views[index]
        output_path = f'{output_dir}/{index}.png'
        #print(output_path)
        
        current_process = subprocess.Popen(
            ["openscad", "-o", output_path, "--camera="+view, "--viewall", "--autocenter", "--imgsize=1024,1024", join(output_dir, file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(current_process)
        #print(index)
        
    for p in processes:
        out, err = p.communicate()
        if 'ERROR' in err:
            #errors = [line for line in err.split('\n') if "WARNING" in line or 'ERROR' in line]
            raise Exception('OpenSCAD code error: '+find_openscad_errors(code)) 
        p.wait()
        
    for index in views:
        view = views[index]
        output_path = f'{output_dir}/{index}.png'
        
        img = Image.open(output_path)
        fullsize = pil_to_bytes(img)
        imgsize = 256, 256
        img.thumbnail(imgsize, Image.Resampling.LANCZOS)
        thumbnail = pil_to_bytes(img)
        
        encoded_imgs.append(fullsize)
        encoded_imgs_sm.append(thumbnail)
    
    return encoded_imgs, encoded_imgs_sm

availableFunctions = [
    {
      "name": "describe",
      "description": "Function to generate descriptions of the model or answer questions about the details",
      "parameters": {
          "type": "object",
          "properties": {
              "question": {
                  "type": "string",
                  "description": "The question the user is asking",
              },
          },
          "required": ["question"],
      },
  },
  {
    "name": "modify",
    "description": "Function to make changes to the model based on the user prompt",
    "parameters": {
        "type": "object",
        "properties": {
            "change": {
                "type": "string",
                "description": "The description of the changes the user wants to make or the model to generate",
            },
        },
        "required": ["change"],
    },
  }          
]

def getDescriptionPrompts(code, text, prevCode, fullCode, partCode, imgs, fullImgs, prevImgs):
    instructions = "describe the visual details such that a blind user could understand it (eg. shape, position, posture, pictures)"
    if len(text) > 0:
        instructions = text
    
    if len(fullCode) > 0:
        instructions = "compare this part of the model in relation to the full model such that a blind user could understand it (eg. spatial position, distance, intersection, size, angle, orientation, side in relation to other parts of the model). Describe how this part affects the model's shape. Only if applicable, mention what operation the part is used in and if it's invisible"
        if len(text) > 0:
            instructions = text
        content = [
                    {"type": "text", "text": "Given the part of a 3D model and its OpenSCAD code, "+instructions+". The part of the model in the code is marked with the comment \"part of the model -->\". Do not mention it was marked with the comment. The first "+str(len(fullImgs))+" images are the full model in different angles with the part of the model highlighted in red and the last "+str(len(imgs))+" images are the part of the model in different angles."},
                    {"type": "text", "text": partCode},
                ]
        
        for img in fullImgs:
            content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}",
                        }})
        
    elif len(prevCode) > 0:
        content = [
                    {"type": "text", "text": "Given the two versions of a 3D model and its OpenSCAD code, with the last "+str(len(imgs))+" images and code referred to as the current model and the first "+str(len(prevImgs))+" images and code referred to as the previous model, describe the changes between the two versions, focusing on the visual details such that a blind user could understand it (eg. shape, position, posture, pictures)."},
                    {"type": "text", "text": prevCode},
                ]
        for img in prevImgs:
            content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}",
                        }})
        content.append({"type": "text", "text": code})
    elif len(code) > 0:
        content = [
                    {"type": "text", "text": "Given the 3D model and its OpenSCAD code, "+instructions},
                    {"type": "text", "text": code},
                ]
    elif len(text) > 0:
        content = [{"type": "text", "text": text}]
    else:
        content = [{"type": "text", "text": "describe how to create a model with openscad"}]
    
    #print(content)
    
    for img in imgs:
        content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                    }})
    
    if len(code) > 0:
        content.append({"type": "text", "text": "You must give a one sentence answer or summary first, followed by more details such that a blind user could understand it. The output should not have formatting since it will be read by a screenreader. Do not mention blind users. The images are of the same model at different angles. Do not mention that there are multiple images. Do not describe each angle separately. The description should be based on the images of the model rather than the code."})
    
    return content


def getModificationPrompts(code, text, imgs):
    if len(code) > 0:
        content = [
            {"type": "text", "text": "Given the OpenScad code, modify the code to "+text+". Output only the modified OpenScad code and nothing else"},
            {"type": "text", "text": code},
        ]
        for img in imgs:
            content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}",
                        }})
    else:
        content = [
            {"type": "text", "text": "Generate the OpenScad code to "+text+". Output only the OpenScad code and nothing else"},
        ]

    return content

@app.route("/api/summarize", methods=["POST"])
def summarize():
    try:
        text = request.form["text"]
        logging.info(f"Received text: {text}")

        template = """
        As a summarizer, here is a paragraph for the blind person to read, but it will take a lot of time for the screen reader to read this paragraph. Please use a simple sentence to restate the main points of the speech so that the blind person can get the most important information in a short time.
{text}
"""
        def gpt_action(text):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": template.format(text=text),
                        },
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(text), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


def addLineNum(code):
    lines = code.split('\n')
    code = ""
    for ln, line in enumerate(lines):
        code = code + "Line " + str(ln) + ": "+line+"\n"
    return code


@app.route("/generate-img", methods=["POST"])
def generate_images():
    global current_process
    try:
        sessionId = request.json.get("sessionId")
        callId = request.json.get("callId")
        code = request.json.get("code")
        prevCode = request.json.get("prevCode")
        fullCode = request.json.get("fullCode")
        imageIndex = request.json.get("imageIndex")
        text = request.json.get("text")
        #if current_process and current_process.poll() is None:
        #    current_process.terminate()
        
        logData = {"sessionId": sessionId, "callId": callId, "function": "generate_images", "timestamps": {"start": time.time()}}

    
        output_dir = "temp/"+str(sessionId)
        os.makedirs(output_dir, exist_ok=True)
        
        if text == "":
            mode = "describe"
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "user", "content": text}
                    ],
                    functions=availableFunctions,
                    function_call="auto"
                )
                #print(response)
                mode = response.choices[0].message.function_call.name
            except:
                pass
        logData["timestamps"]["getMode"] = time.time()
        
        changes = ""
        if prevCode != "" and prevCode != code:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    #stream=True,
                    messages=[
                        {
                        "role": "user",
                        "content": [
                                {"type": "text", "text": 'Given the previous OpenSCAD code followed by the current OpenSCAD code, output the list of chunks of code that were added, deleted or changed in the format [{"startLine": <the first line number of the chunk in the current code, or -1>, "endLine": <the last line number of the chunk>, "description": <description of what changed>}]. Output only JSON and nothing else'},
                                {"type": "text", "text": "Previous code:\n\n"+addLineNum(prevCode)},
                                {"type": "text", "text": "Current code:\n\n"+addLineNum(code)},
                            ],
                        }
                    ],
                )
                changes = response.choices[0].message.content.replace('```json', '').replace('```', '')
                json.loads(changes)
                #print(changes)
            except:
                pass
        logData["timestamps"]["getChanges"] = time.time()
        
        
        views = {
            "display": "50,50,50,60,30,-210,300",   
        }

        encoded_imgs, encoded_imgs_sm = gen_image(views, code, output_dir)
        if encoded_imgs is None:
            return jsonify(error=f"Failed to generate image {index}"), 500
        
        
        if len(fullCode) > 0:
            _, encoded_imgs_full = gen_image(views, fullCode, output_dir)
        else:
            encoded_imgs_full = [""]
            
        logData["timestamps"]["getImg"] = time.time()
        with open('log.txt', 'a') as f:
            f.write(json.dumps(logData)+'\n')
        
        
        return jsonify({"message": "Images generated successfully", "mode": mode, "changes": changes, "image": encoded_imgs[0], "thumbnail": encoded_imgs_sm[0], "fullImg": encoded_imgs_full[0]})
    except Exception as e:
        print(f"Execution error: {e}")
        logData["error"] = str(e)
        with open('log.txt', 'a') as f:
            f.write(json.dumps(logData)+'\n')
        if 'OpenSCAD code error: ' in str(e):
            return jsonify({'message': "OpenSCAD code error", 'error': str(e).replace('OpenSCAD code error: ','')})
        return jsonify({'message': "Failed to generate images.", 'error': str(e)}), 500

@app.route("/api/describe", methods=["POST"])
def describe():
    try:
        sessionId = request.json.get("sessionId")
        callId = request.json.get("callId")
        text = request.json.get("text")
        code = request.json.get("code")
        mode = request.json.get("mode")
        prevCode = request.json.get("prevCode")
        fullCode = request.json.get("fullCode")
        partCode = request.json.get("partCode")
        logging.info(f"Received text: {text}")
        
        logData = {"sessionId": sessionId, "callId": callId, "function": "describe", "prompt": text, "mode": mode, "code": code, "timestamps": {"start": time.time()}}
        
        
        if prevCode == code:
            prevCode = ""
        if fullCode == code:
            fullCode = ""
            
        imgs = []
        prevImgs = []
        fullImgs = []
        views = [
            "50,50,50,60,30,-210,300", 
            "0,0,0,0,0,0,200",          
            "0,0,-50,180,0,180,200",     
            "-50,0,0,90,0,0,200",      
            "50,0,0,-90,180,0,200",      
            "0,50,0,90,0,90,200",      
            "0,-50,0,90,0,-90,200"    
        ]
        #if mode == "modify":
        #    views = ["50,50,50,60,30,-210,300"]
        views = dict(enumerate(views))
        output_dir = "temp/"+str(sessionId)
        os.makedirs(output_dir, exist_ok=True)

        if len(fullCode) > 0:
            _, fullImgs = gen_image(views, fullCode, output_dir)
        if len(prevCode) > 0:
            _, prevImgs = gen_image(views, prevCode, output_dir)
        if len(code) > 0:
            _, imgs = gen_image(views, code, output_dir)
            
        logData["timestamps"]["genViews"] = time.time()

        if text == "":
            content = getDescriptionPrompts(code, text, prevCode, fullCode, partCode, imgs, fullImgs, prevImgs)
        else:
            if mode == "modify":
                if len(fullCode) > 0:
                    code = fullCode
                content = getModificationPrompts(code, text, imgs)
            else:
                content = getDescriptionPrompts(code, text, prevCode, fullCode, partCode, imgs, fullImgs, prevImgs)
        
        def gpt_action(content, mode):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    #stream=True,
                    messages=[
                        {
                        "role": "user",
                        "content": content,
                        }
                    ],
                )
                return completion.choices[0].message.content
                
                """
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
                """
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    return "Error: " + str(e) 
        
        response = gpt_action(content, mode) 
        
        logData["response"] = response
        logData["timestamps"]["genDesc"] = time.time()
        with open('log.txt', 'a') as f:
            f.write(json.dumps(logData)+'\n')
            
        return Response(response, mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        logData["error"] = e
        with open('log.txt', 'a') as f:
            f.write(json.dumps(logData)+'\n')
        return jsonify({'message': "Internal server error", 'error': str(e)}), 500



@app.route("/api/callGPT", methods=["POST"])
def callGPT():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        # Save the image and encode it
        image_path = "./static/img/temp.jpg"
        image.save(image_path)
        img_url = "data:image/jpeg;base64," + encode_image(image_path)

        template = """
Code2Fab is a system that helps the blind to use OpenSCAD to model for 3D printing. You are an accessible 3D printing expert for the blind and work for Code2Fab. Your primary role is to empower blind users to create and understand 3D models using OpenSCAD.

**Important Considerations:**

* This is a single interaction, so you must provide a comprehensive and helpful response based on the user's initial question.
* Blind users may not be able to provide additional context, so be prepared to ask clarifying questions or offer multiple potential interpretations of their question.
* Tailor your language to be clear, concise, and accessible to users of screen readers and braille displays.

**User's Question:** "{text}"

**Model's OpenSCAD Code:** 
***Report Begins*** 
{code}
***Report Ends*** 

**Your Response Should Include:**

1. **Direct Answer:** If possible, provide a clear and concise answer to the user's question based on the OpenSCAD code.
2. **Clarification Questions:** If the question is ambiguous, ask specific questions to better understand the user's needs.
3. **Multiple Interpretations:** If the question could be interpreted in different ways, offer multiple potential answers or explanations.
4. **Additional Guidance:** If relevant, provide suggestions for troubleshooting, design improvements, or alternative approaches.

**Example Responses:**

* **Direct Answer:** "Based on the code, your model is a cube with sides of 10mm each."
* **Clarification Question:** "Could you clarify which part of the code you'd like me to explain? Are you interested in the `cube()` function or the `translate()` function?"
* **Multiple Interpretations:** "This line of code could either create a cylinder with a radius of 5mm or a sphere with a diameter of 5mm. Which shape are you trying to create?"
* **Additional Guidance:** "To make your cube larger, you could increase the values inside the `cube()` function. For example, `cube([20,20,20]);` would create a cube with sides of 20mm."
        """

        def gpt_action(img_url):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": template.format(text=text, code=code)},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                            },
                            },
                        ],
                        }
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(img_url), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500
    

@app.route("/api/match", methods=["POST"])
def match():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        # Save the image and encode it
        image_path = "./static/img/temp.jpg"
        image.save(image_path)
        img_url = upload_image(image_path)

        template = f"""
You will first ask the model descriptor to describe what the model consists of and what it looks like, and then the code interpreter will work with the model descriptor to match the different parts of the model to the corresponding code:
<img {img_url}>.

Extra requirement: [{text}]
***Code Begins***
'''openscad'''
{code}
'''openscad'''
***Code Ends***

Use the follow format for output:
***Report Begins***
##Direct Reply for user's input##
[Insert the direct reply for user's input. DO NOT return this section if there isn't any extra requirement.]

##Description of the model##
[Insert the description of the model here, highlighting key elements.]

##Summary of the model##
[Insert the summary of the model here, contains all the components.]

##Codes##
"Code1", [The corresponding part in the model], 
[content of Code1]
"Code2", [The corresponding part in the model], 
[content of Code1]
...
***Report Ends***
"""
        user_proxy.initiate_chat(manager, message=template)

        message = code_interpreter.last_message()['content']

        try:
            return message
        except Exception as e:
            logging.error(f"Error processing GPT action: {e}")
            error_message = "Error: " + str(e)
            return error_message

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/analysis", methods=["POST"])
def analysis():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        template = """
As a code interpreter, you will receive a set of OpenSCAD code and analyze the code for a blind user to understand.
Given the Openscad code, you will analyze the code and provide a detailed description of the code, highlighting the key elements and code structure. After that, you will evaluate the code, highlighting the strengths and weaknesses. 
***Code Begins***
'''openscad'''
{code}
'''openscad'''
***Code Ends***

Use the follow format for output:
***Report Begins***

##Description of the openscad code##
[Insert the description of the openscad here, highlighting key elements and code structure.]

##Summary of the code##
[Insert the summary of the code here, contains all the components.]

##Evaluation of the code##
[Insert the evaluation of the code here, highlighting the strengths and weaknesses.]

##Codes##
"Code1", [Function of the code],[Suggestions for improvement] 
[content of Code1]
"Code2", [Function of the code],[Suggestions for improvement]
[content of Code1]
...
***Report Ends***
        """

        def gpt_action(text, code):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": template.format(text=text, code=code),
                        },
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(text, code), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/improve", methods=["POST"])
def improve():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        template = """
As a professional code reviewer, you will receive a set of OpenSCAD code and provide suggestions for improving the code for a blind user to improve the code.
{text}
***Code Begins***
'''openscad'''
{code}
'''openscad'''
***Code Ends***

Follow the template below to output the result:
***Template Begins***
##Suggestions for improving the code##
[Insert the suggestions for improving the code here, highlighting key elements and code structure.]

##Evaluation of the code##
[Insert the evaluation of the code here, highlighting the strengths and weaknesses.]

##Details for Codes' improvement##
"Code1", [Function of the code],[Suggestions for improvement]
Original Code: [content of Code1]
Improved Code: [Improved content of Code1]

"Code2", [Function of the code],[Suggestions for improvement]
Original Code: [content of Code2]
Improved Code: [Improved content of Code2]
...
***Template Ends***
"""
        def gpt_action(text, code):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": template.format(text=text, code=code),
                        },
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(text, code), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/api/create", methods=["POST"])
def create():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")
  
        template = """
You are an OpenSCAD expert specializing in accessible code generation for individuals who are blind or visually impaired.  Your primary goal is to translate user descriptions of 3D models into functional, efficient, and accessible OpenSCAD code within a single interaction.

Core Responsibilities:

1.  Comprehensive Requirement Analysis:
- Actively listen to the user's description of their desired 3D model.
- Focus on understanding their vision, including the model's overall shape, dimensions, features, and any specific functional requirements.
- If necessary, politely request clarifying details or suggest alternative approaches to ensure a clear understanding of the project scope.

2.  Accessible Code Generation:
- Transform the user's description into precise, well-structured OpenSCAD code that adheres to industry best practices.
- Prioritize accessibility by:
- Employing clear, descriptive variable names and comments.
- Implementing consistent indentation and formatting for seamless navigation with screen readers.
- Utilizing modules and functions to enhance code organization and reusability.

3.  Proactive Guidance and Optimization:
- Proactively identify and address potential challenges or ambiguities in the user's model description.
- Offer expert suggestions to refine the model's design, enhance functionality, or optimize code efficiency.
- Provide constructive feedback and alternative solutions if errors or inconsistencies are detected in the user's input.
- Empower users to expand their OpenSCAD knowledge and skills through concise, informative guidance.

Guiding Principles:

- **Professional Communication:** Maintain a courteous, respectful, and professional tone in all interactions.
- **Technical Clarity:** Communicate technical concepts in a clear, concise manner, avoiding unnecessary jargon.
- **User Empowerment:** Foster a collaborative environment that encourages user participation, experimentation, and skill development.
- **Accessibility Focus:** Ensure generated code and all communication are fully accessible to individuals using assistive technologies.
- **Continuous Improvement:** Actively seek user feedback to refine your code generation process and enhance the overall user experience.

User‘s requirement: "{text}".


Follow the template below to output the result:
*Template Begins*
```
***OpenSCAD Model Generation Report***

**User's Project Description:** [Concisely reiterate the user's model description.]

**Generated OpenSCAD Code:**
```openscad
[Insert the complete, accessible, and optimized OpenSCAD code here.]
```

**Technical Analysis and Recommendations:**
* [Provide a brief analysis of the code's structure, key components, and how it fulfills the user's requirements.]
* [Offer specific recommendations on potential modifications or enhancements to the code, tailored to the user's needs and skill level.]
* [Highlight any potential limitations or areas for further exploration based on the project scope and OpenSCAD capabilities.]

***OpenSCAD Model Generation Report Ends***

*Template Ends*
"""
        def gpt_action(text, code):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": template.format(text=text, code=code),
                        },
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(text, code), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
    