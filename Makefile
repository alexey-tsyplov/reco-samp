POETRY = poetry
MANAGER = poetry run
CODE = recosamp tests

install: # Install module in venv
	$(POETRY) lock
	$(POETRY) install

lint: # Lint code
	$(MANAGER) ruff check $(CODE)
	$(MANAGER) pylint $(CODE)
	$(MANAGER) mypy $(CODE)

format: # Formats all files
	$(MANAGER) ruff check --fix $(CODE)
	$(MANAGER) ruff format $(CODE)

test: # Runs pytest
	$(MANAGER) pytest
