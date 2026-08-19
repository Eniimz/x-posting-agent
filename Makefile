.PHONY: eval graders

eval:
	PYTHONPATH=. .venv/bin/python evals/eval.py

graders:
	PYTHONPATH=. .venv/bin/python evals/graders.py
