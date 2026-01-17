# A11yShape

## Starting the server

Here are the instructions for setting up the backend server on your own computer:

Install OpenSCAD (https://openscad.org/downloads.html) and make sure the folder containing openscad.exe is added to the system PATH environment variable.

Optional: Install Ngrok from https://ngrok.com/download (only needed for remote access)

In commandline, run:

```
ngrok config edit
```

Set the contents of ngrok.yml to be the same as the ngrok.yml file from this repo

Run:

```
ngrok start --all
```

In the A11yShape folder run:

```
pip install -r requirements.txt
python app.py
```

Local-only mode

- Open http://127.0.0.1:3000 or http://localhost:3000 in your browser.
- The frontend calls the local origin for API requests by default.
- To use ngrok, set A11YSHAPE_API_BASE to your ngrok https URL and restart Flask.

Test the backend server at https://livid-memorisingly-lavonne.ngrok-free.dev/ (only when ngrok is enabled)

If it's running successfully, it should display:

```
Hello from Flask6!
```

The frontend is always running at https://a11yshape.pages.dev/. It will use the local API base by default when served from Flask.

## Updating changes to the server

In the A11yShape folder on the server, run:

```
git pull
```

Stop the frontend and backend and restart with:

```
python app.py
python -m http.server 9000
```

