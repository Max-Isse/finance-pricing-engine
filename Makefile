.PHONY: setup test gate train recalibrate simulate-drift serve run clean

setup:
	pip install -r requirements.txt

test:
	python -m pytest -q

gate:
	python pipelines/quality_gate.py --simulate

train:
	python pipelines/weekly_retrain.py

recalibrate:
	python pipelines/intraday_recalibration.py

simulate-drift:
	python monitoring/drift_monitors.py

serve:
	uvicorn src.service.app:app --host 0.0.0.0 --port 8000

run:
	docker-compose up --build -d

clean:
	docker-compose down -v
	rm -rf data/*.csv artifacts
