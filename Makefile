install:
	pip install -r requirements.txt

train:
	python scripts/train.py

test:
	pytest