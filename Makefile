.PHONY: prepare retrieval candidates ranker evaluate batch

export PYTHONPATH=src

prepare:
	python -m pipelines.01_prepare_data

retrieval:
	python -m pipelines.02_train_retrieval

candidates:
	python -m pipelines.03_generate_candidates --split train
	python -m pipelines.03_generate_candidates --split val
	python -m pipelines.03_generate_candidates --split test

ranker:
	python -m pipelines.04_train_ranker

evaluate:
	python -m pipelines.05_evaluate

batch:
	python -m pipelines.06_batch_recommend --split test --top-k 10
