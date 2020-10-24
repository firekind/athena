ifeq ($(OS),Windows_NT)
	ENV_NAME=win_env
	PYTHON_PATH=./$(ENV_NAME)/Scripts/python.exe
else
	ENV_NAME=lin_env
	PYTHON_PATH=./$(ENV_NAME)/bin/python
endif

venv:
	@virtualenv $(ENV_NAME)
	@$(PYTHON_PATH) -m pip install -r requirements.txt
	@$(PYTHON_PATH) -m pip install -e .