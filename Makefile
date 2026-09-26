.PHONY: setup data pipeline audit train explain monitor retrain api dashboard test docker-up docker-down

setup:
	python -m pip install -r requirements-dev.txt

data:
	python scripts/copy_local_data.py

pipeline:
	python scripts/run_pipeline.py

audit:
	python scripts/run_audit.py

train:
	python scripts/train_champion.py

explain:
	python scripts/run_explainability.py

monitor:
	python scripts/run_monitoring.py

retrain:
	python scripts/retrain_if_needed.py

api:
	uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	python services/dashboard/app.py

test:
	pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down
