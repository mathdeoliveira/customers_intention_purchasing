train:
	python src/train.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

mlflow-shutdown:
	fuser -k 5000/tcp

quality_checks:
	isort src/
	black src/