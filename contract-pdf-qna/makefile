include .env
export



cacert.pem:
	@echo 'cacert.pem doesnt exsist' && false

run: cacert.pem bigquery.json
	uv run python app.py

init:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv init .
	uv add -r requirements.txt
attu:
	docker run --name attu -p 8000:3000 -e MILVUS_URL=$(MILVUS_HOST):19530 zilliz/attu:latest || docker start attu
jaeger:
	docker run --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/jaeger:latest

event_test:
	. .venv/bin/activate; python3 -c "from tests import test_handle_transcript_event"

test_time:
	. venv/bin/activate; python3 -c "from tests import tests_measure"
