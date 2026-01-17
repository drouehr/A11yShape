"""
This code defines a Flask application that serves as an API for generating STL files, saving and loading OpenSCAD code files, and performing model description and code matching tasks using OpenAI's gpt-5 models.

The API endpoints include:
- /generate-stl: Accepts a POST request with OpenSCAD code and generates an STL file.
- /save-code: Accepts a POST request with OpenSCAD code and saves it as a file.
- /get-files: Returns a list of saved code files.
- /load-code/<file_name>: Returns the content of a specific code file.
- /api/describe: Accepts a POST request with text, code, and an image, and uses gpt-5 to generate a description of the 3D model.
- /api/match: Accepts a POST request with text, code, and an image, and uses gpt-5 to match the different parts of the 3D model to the corresponding code.
- /api/analysis: Accepts a POST request with text and code, and uses gpt-5 to analyze the OpenSCAD code.
- /api/improve: Accepts a POST request with text and code, and uses gpt-5 to provide suggestions for improving the code.
The code also includes configuration settings for the gpt-5 models, agent definitions for model descriptor, code interpreter, and user proxy, and a Flask route for serving the index.html file.

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
import shutil

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENSCAD_PATH = (os.environ.get("OPENSCAD_PATH") or "").strip() or "openscad"
resolved_openscad = shutil.which(OPENSCAD_PATH)
if resolved_openscad:
    OPENSCAD_PATH = resolved_openscad
logging.info(f"OPENSCAD_PATH resolved to: {OPENSCAD_PATH!r}")

from PIL import Image
import io
import json
import time

LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

logging.getLogger("werkzeug").setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logging.getLogger().info("LOG_LEVEL=%s", LOG_LEVEL)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing OpenAI API key. Set OPENAI_API_KEY in your environment/.env"
    )

base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None

MODEL_NANO = os.environ.get("OPENAI_MODEL_LIGHT", "gpt-5-nano")
MODEL_MINI = os.environ.get("OPENAI_MODEL_HEAVY", "gpt-5-mini")

client = OpenAI(api_key=api_key, base_url=base_url)

"""AutoGen Config"""
config_llm_4v = [{"model": MODEL_NANO, "api_key": api_key}]

config_llm_4 = [{"model": MODEL_NANO, "api_key": api_key}]

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


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
PORT = int(os.environ.get("PORT", 3000))
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

A11YSHAPE_API_BASE = (
    os.environ.get("A11YSHAPE_API_BASE")
    or os.environ.get("NGROK_URL")
    or ""
).strip()

@app.route("/")
def index():
    return send_from_directory(APP_ROOT, "index.html")

@app.route("/index_Chinese.html")
def index_chinese():
    return send_from_directory(APP_ROOT, "index_Chinese.html")

@app.route("/config.js")
def client_config():
    api_base = A11YSHAPE_API_BASE.replace("\\", "\\\\").replace("\"", "\\\"")
    body = f'window.__A11YSHAPE_API_BASE__ = "{api_base}";\n'
    return Response(body, mimetype="text/javascript")

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
        model=MODEL_NANO,
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
    """Render OpenSCAD code to one or more PNGs and return base64 images."""
    code = code.encode("ascii", errors="ignore").decode().lower()

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("gen_image cwd=%s", os.getcwd())
        logging.debug("gen_image output_dir=%s", output_dir)
        logging.debug("OPENSCAD_PATH=%s", OPENSCAD_PATH)
        logging.debug("PATH=%s", os.environ.get("PATH", ""))
        logging.debug("which(OPENSCAD_PATH)=%s", shutil.which(OPENSCAD_PATH))

    file = "model.scad"
    with open(join(output_dir, file), "w", encoding="utf-8") as f:
        f.write(code)

    encoded_imgs = []
    encoded_imgs_sm = []

    processes = []
    for index in views:
        view = views[index]
        output_path = os.path.join(output_dir, f"{index}.png")

        input_scad_path = os.path.join(output_dir, file)
        cmd = [
            OPENSCAD_PATH,
            "-o",
            output_path,
            "--camera=" + view,
            "--viewall",
            "--autocenter",
            "--imgsize=1024,1024",
            input_scad_path,
        ]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("OpenSCAD command: %s", cmd)
            logging.debug("input_scad_path exists=%s", os.path.exists(input_scad_path))
            logging.debug("output_path=%s", output_path)

        try:
            current_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as e:
            raise Exception(
                "OpenSCAD executable not found. "
                "Set OPENSCAD_PATH in your .env to the full path of openscad.exe, e.g. "
                "OPENSCAD_PATH=C:\\Program Files\\OpenSCAD\\openscad.exe. "
                f"Currently OPENSCAD_PATH='{OPENSCAD_PATH}'."
            ) from e

        processes.append(current_process)

    for p in processes:
        out, err = p.communicate()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("OpenSCAD returncode: %s", p.returncode)
            logging.debug("OpenSCAD stdout: %s", (out or "").strip())
            logging.debug("OpenSCAD stderr: %s", (err or "").strip())
        if err and "ERROR" in err:
            raise Exception("OpenSCAD code error: " + find_openscad_errors(code))
        p.wait()

    for index in views:
        output_path = os.path.join(output_dir, f"{index}.png")

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Checking output image exists=%s path=%s", os.path.exists(output_path), output_path)

        if not os.path.exists(output_path):
            raise Exception(f"OpenSCAD did not produce expected output file: {output_path}")

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
      "description": "Use when the user asks to describe, analyze, or explain the current model or code. Do not modify code.",
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
    "description": "Use when the user requests changes, additions, removals, or creation of geometry. Return OpenSCAD code only.",
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
    instructions = "Describe only the physical geometry of the model (shape, size, proportions, relative positions, intersections, symmetry, and orientation). Avoid background, color, lighting, or rendering details. Do not explain how to create the model unless explicitly asked."
    if len(text) > 0:
        instructions = text
    
    if len(fullCode) > 0:
        instructions = "Compare this part to the full model in terms of spatial position, scale, intersections, and how it changes the overall shape. Avoid background, color, lighting, or rendering details. Do not explain how to create the model unless explicitly asked."
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
                    {"type": "text", "text": "Given the two versions of a 3D model and its OpenSCAD code, with the last "+str(len(imgs))+" images and code referred to as the current model and the first "+str(len(prevImgs))+" images and code referred to as the previous model, describe the changes between the two versions focusing on physical geometry (shape, size, proportions, relative positions, intersections, symmetry, and orientation). Avoid background, color, lighting, or rendering details."},
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
        content = [{"type": "text", "text": "Describe the model's physical geometry in plain language."}]
    
    #print(content)
    
    for img in imgs:
        content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                    }})
    
    if len(code) > 0:
        content.append({"type": "text", "text": "Start with a one-sentence summary, then provide concise details. Output plain text only (no formatting). Do not mention the user or how to create the model. Do not mention multiple images or viewing angles. Describe only the model's physical geometry and spatial relationships."})
    
    return content


def getModificationPrompts(code, text, imgs):
    if len(code) > 0:
        content = [
            {"type": "text", "text": "Modify the OpenSCAD code to satisfy: "+text+". Return ONLY the modified OpenSCAD code with no commentary, no backticks, and no extra text. Preserve unchanged code where possible."},
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
            {"type": "text", "text": "Generate OpenSCAD code that satisfies: "+text+". Return ONLY the OpenSCAD code with no commentary, no backticks, and no extra text."},
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
                    model=MODEL_NANO,
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


MODIFY_KEYWORDS = [
    "add",
    "remove",
    "delete",
    "change",
    "modify",
    "make",
    "create",
    "generate",
    "insert",
    "cut",
    "drill",
    "hole",
    "subtract",
    "difference",
    "union",
    "intersect",
    "translate",
    "rotate",
    "scale",
    "resize",
    "taller",
    "shorter",
    "wider",
    "narrower",
    "bigger",
    "smaller",
    "thicker",
    "thinner",
    "through",
    "cylinder",
    "cube",
    "sphere",
    "extrude",
    "fillet",
    "chamfer",
]

DESCRIBE_KEYWORDS = [
    "describe",
    "explain",
    "analyze",
    "analysis",
    "summary",
    "summarize",
    "what is",
    "what's",
    "details",
    "how does",
]


def infer_mode_from_text(text):
    lowered = (text or "").lower()
    if not lowered.strip():
        return None
    if any(keyword in lowered for keyword in MODIFY_KEYWORDS):
        return "modify"
    if any(keyword in lowered for keyword in DESCRIBE_KEYWORDS):
        return "describe"
    return None


@app.route("/generate-img", methods=["POST"])
def generate_images():
    global current_process
    try:
        payload = request.get_json(silent=True) or {}
        sessionId = payload.get("sessionId")
        callId = payload.get("callId")
        code = payload.get("code") or ""
        prevCode = payload.get("prevCode") or ""
        fullCode = payload.get("fullCode") or ""
        imageIndex = payload.get("imageIndex")
        text = payload.get("text") or ""
        requested_mode = (payload.get("mode") or "").strip().lower()

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("/generate-img cwd=%s", os.getcwd())
            logging.debug("/generate-img env OPENSCAD_PATH=%s", OPENSCAD_PATH)
            logging.debug("/generate-img env PATH=%s", os.environ.get("PATH", ""))
            logging.debug(
                "/generate-img request payload: %s",
                json.dumps(
                    {
                        "sessionId": sessionId,
                        "callId": callId,
                        "imageIndex": imageIndex,
                        "text_len": len(text),
                        "code_len": len(code),
                        "prevCode_len": len(prevCode),
                        "fullCode_len": len(fullCode),
                    },
                    ensure_ascii=False,
                ),
            )

        #if current_process and current_process.poll() is None:
        #    current_process.terminate()

        logData = {"sessionId": sessionId, "callId": callId, "function": "generate_images", "timestamps": {"start": time.time()}}

    
        output_dir = "temp/"+str(sessionId)
        os.makedirs(output_dir, exist_ok=True)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("/generate-img output_dir created=%s", output_dir)
        
        mode = "describe"
        if requested_mode in {"describe", "modify"}:
            mode = requested_mode
        else:
            heuristic_mode = infer_mode_from_text(text)
            if heuristic_mode:
                mode = heuristic_mode
            elif text != "":
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NANO,
                        messages=[{"role": "user", "content": text}],
                        functions=availableFunctions,
                        function_call="auto",
                    )

                    fn_call = getattr(response.choices[0].message, "function_call", None)
                    if fn_call and getattr(fn_call, "name", None):
                        mode = fn_call.name
                except Exception as e:
                    logging.warning(f"/generate-img mode detection failed; defaulting to describe: {e}")

        logData["timestamps"]["getMode"] = time.time()
        
        changes = ""
        if prevCode != "" and prevCode != code:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NANO,
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
        logging.exception("/generate-img failed")
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
                    model=MODEL_NANO,
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
                    model=MODEL_NANO,
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
                    model=MODEL_NANO,
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
                    model=MODEL_NANO,
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
                    model=MODEL_NANO,
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
    