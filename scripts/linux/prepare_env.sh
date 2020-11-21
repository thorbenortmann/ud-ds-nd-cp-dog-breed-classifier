#!/bin/bash

# Set the working folder to the project folder
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPTS_DIR" && cd ..

# Install the virtualenv package
pip3 install virtualenv

# Create the virtual env
if test -f "venv\Scripts\activate"; then
    echo "Virtual environment venv is already created!"
else
    virtualenv -p python3 venv
fi

# Activate the virtual env
source venv/bin/activate

# Install python package management packages
pip install pip setuptools wheel

# Install python packages of the application
pip install -e .
