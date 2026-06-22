.PHONY: run test pipeline install clean

run:
	uvicorn api_main:app --port 8001 --reload

test:
	python -m pytest tests/ -v

install:
	pip install -r requirements.txt

pipeline:
	python src/module1_ingest_parse.py data/raw/HDFS_sample_1pct.log --dataset hdfs --max-lines 100000
	python src/module2_session_anomaly.py data/processed/HDFS_sample_1pct_structured.csv --dataset hdfs --label-path data/raw/anomaly_label.csv --contamination 0.03
	python src/module3_embed_index.py data/processed/HDFS_sample_1pct_anomalies.json --dataset hdfs --model all-MiniLM-L6-v2
	python src/module4_rag_analysis.py data/processed/HDFS_sample_1pct_anomalies.json --dataset hdfs --max-sessions 5

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
