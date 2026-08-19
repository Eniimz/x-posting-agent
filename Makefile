PYTHON ?= .venv/bin/python

.PHONY: eval graders

eval:
	PYTHONPATH=agent:. $(PYTHON) evals/eval.py

graders:
	PYTHONPATH=agent:. $(PYTHON) evals/graders.py
