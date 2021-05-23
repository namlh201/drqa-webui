import os
import sys
import json
import logging
from flask import Flask, render_template, request

from pydantic import BaseSettings, BaseModel
class Settings(BaseSettings):
    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"

settings = Settings()

if settings.USE_NGROK:
  
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
    from pyngrok import ngrok

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port)

    logging.warning("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    
from services import Services

def app(retriever='tfidf'):
    app = Flask(__name__)
    services = Services(retriever)

    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route("/query", methods=["POST"])
    def query():
        data = request.json
        answers = services.process(question=data['question'])
        return json.dumps(answers)

    return app