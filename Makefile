.PHONY: setup run test simulate-drift clean

setup:
	pip install -r requirements.txt

run:
	docker-compose up --build -d

test:
	python -m unittest discover -s src/tests -p "*_test.py"

simulate-drift:
	@echo "Injecting artificial volatility adjustments into active stream..."
	python monitoring/drift_monitors.py

clean:
	docker-compose down -v
	rm -rf data/*.csv