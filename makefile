ifeq ($(OS),Windows_NT)
	PYTHON_PATH=./venv/Scripts/python.exe
else
	PYTHON_PATH=./venv/bin/python
endif

venv:
	@virtualenv venv
	@$(PYTHON_PATH) -m pip install -r requirements.txt
	@$(PYTHON_PATH) -m pip install -e .