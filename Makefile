PYTHON ?= python
PIP ?= pip

REQ=requirements.txt

.PHONY: setup fmt lint test ci run inspect cov

setup:
	$(PIP) install -r $(REQ)

fmt:
	@echo "(placeholder) add black/ruff later"

lint:
	@echo "(placeholder) lint step - add ruff/mypy later"

test:
	PYTHONPATH=. pytest -q

cov:
	PYTHONPATH=. pytest --maxfail=1 --cov=src --cov-report=term-missing

ci: fmt lint cov

run:
	$(PYTHON) train.py --num-envs 1 --total-updates 30 --log-interval 5 --no-compile --scripted-forward-frames 64 --metrics-path tmp_metrics.jsonl

inspect:
	$(PYTHON) scripts/inspect_metrics.py --path tmp_metrics.jsonl --tail 30 || echo "请先执行 make run 生成 tmp_metrics.jsonl"
