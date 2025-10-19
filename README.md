# A11yShape

## Starting the server

Here are the instructions for setting up the backend server on your own computer:

Install Ngrok from https://ngrok.com/download

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
python app.py
```

Test the backend server at https://livid-memorisingly-lavonne.ngrok-free.dev/

If it's running successfully, it should display:

```
Hello from Flask6!
```

The frontend is always running at https://a11yshape.pages.dev/ and is already configured to access the backend server.

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

