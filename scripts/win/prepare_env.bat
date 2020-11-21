@ECHO OFF

:: Set the working folder to the root project folder
for %%B in (%~dp0\.) do set SCRIPTS_DIR=%%~dpB
cd %SCRIPTS_DIR% && cd ..

:: Install the virtualenv package
pip install virtualenv

:: Create the virtual env
if exist venv\Scripts\activate.bat (
    ECHO "Virtual environment venv is already created!"
) else (
    virtualenv -p python venv
)

:: Activate the virtual env
CALL venv\Scripts\activate.bat

:: Install python package management packages
pip install pip setuptools wheel

:: Install python packages of the application
pip install -e .
