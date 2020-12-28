#!/bin/bash

# Set the working folder to the project folder
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPTS_DIR" && cd ../../

# Configure the flask app
export FLASK_DEBUG=0
export FLASK_RUN_PORT=5000
export FLASK_ENV=development
export FLASK_APP=dog_breed_classifier/app/app.py

# Activate the virtual env
source venv/bin/activate

:: Run the flask app
python -m flask run --eager-loading
