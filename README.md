# A11yShape

## Starting the server

Here are the instructions for setting up a server on your own computer:

download the zip file in https://ngrok.com/download

in command prompt, navigate to the folder containing ngrok.exe

Run:

```
ngrok.exe config edit
```

Set the contents of ngrok.yml to be the same as the ngrok.yml file in this repo

Change the hostname of frontend and backend

Run:

```
ngrok.exe start --all
```

make sure the PORT in app.py matches the addr of backend in ngrok.yml

make sure the serverUrl in index.html matches the hostname of backend in ngrok.yml

in the code2fab folder run:

```
python app.py
python -m http.server 9000
```

## Updating changes to the server

In the code2fab folder on the server, run:

```
git pull
```

Stop the frontend and backend and restart with:

```
python app.py
python -m http.server 9000
```

