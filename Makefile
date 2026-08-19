.PHONY: eval graders

eval:
	PYTHONPATH=agent:. .venv/bin/python evals/eval.py

graders:
	PYTHONPATH=agent:. .venv/bin/python evals/graders.py
