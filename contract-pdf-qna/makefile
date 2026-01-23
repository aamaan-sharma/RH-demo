
init:
	pyenv local
	python3 -m venv venv
	. venv/bin/activate; python3 -m pip install -r requirements.txt
run:
	. venv/bin/activate; python3 app.py
attu:
	docker run -p 8000:3000 -e MILVUS_URL=$(MILVUS_HOST):19530 zilliz/attu:latest
jaeger:
	docker run -p 16686:16686 -p 4318:4318 jaegertracing/jaeger:latest
