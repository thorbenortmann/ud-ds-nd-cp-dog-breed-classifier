@ECHO OFF

:: Set the working folder to the root project folder
for %%B in (%~dp0\.) do set SCRIPTS_DIR=%%~dpB
cd %SCRIPTS_DIR% && cd ..

:: Configure the flask app
SET FLASK_DEBUG=true
SET FLASK_RUN_PORT=5000
SET FLASK_ENV=development
SET FLASK_APP=dog_breed_classifier/app/app.py

:: Activate the virtual env
CALL venv\Scripts\activate.bat

:: Run the flask app
python -m flask run
