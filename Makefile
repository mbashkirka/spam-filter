.PHONY: quality test m0-quality m0-test m0-smoke m1-quality m1-test m1-data m1-smoke m1-all m2-quality m2-test m2-train m2-smoke m2-all m3-quality m3-test m3-train m3-smoke m3-all m4-quality m4-test m4-train m4-smoke m4-all m5-quality m5-test m5-train m5-smoke m5-all m6-quality m6-test m6-data m6-eval m6-smoke m6-all m7-quality m7-test m7-run m7-smoke m7-all m8-quality m8-test m8-freeze m8-smoke m8-api-smoke m8-serve m8-all m9-quality m9-test m9-benchmark m9-release m9-smoke m9-api-smoke m9-serve m9-all

quality: m9-quality

test: m9-test

m0-quality:
	python -m ruff check .

m0-test:
	python -m pytest -q

m0-smoke:
	python scripts/m0_smoke.py

m1-quality:
	python -m ruff check .

m1-test:
	python -m pytest -q

m1-data:
	python scripts/m1_prepare.py

m1-smoke:
	python scripts/m1_smoke.py

m1-all: m1-quality m1-test m1-data m1-smoke

m2-quality:
	python -m ruff check .

m2-test:
	python -m pytest -q

m2-train:
	python scripts/m2_train.py

m2-smoke:
	python scripts/m2_smoke.py

m2-all: m2-quality m2-test m1-smoke m2-train m2-smoke

m3-quality:
	python -m ruff check .

m3-test:
	python -m pytest -q

m3-train:
	python scripts/m3_train.py

m3-smoke:
	python scripts/m3_smoke.py

m3-all: m3-quality m3-test m1-smoke m3-train m3-smoke

m4-quality:
	python -m ruff check .

m4-test:
	python -m pytest -q

m4-train:
	python scripts/m4_train.py

m4-smoke:
	python scripts/m4_smoke.py

m4-all: m4-quality m4-test m1-smoke m3-smoke m4-train m4-smoke

m5-quality:
	python -m ruff check .

m5-test:
	python -m pytest -q

m5-train:
	python scripts/m5_train.py

m5-smoke:
	python scripts/m5_smoke.py

m5-all: m5-quality m5-test m1-smoke m3-smoke m4-smoke m5-train m5-smoke


m6-quality:
	python -m ruff check .

m6-test:
	python -m pytest -q

m6-data:
	python scripts/m6_prepare.py

m6-eval:
	python scripts/m6_eval.py

m6-smoke:
	python scripts/m6_smoke.py

m6-all: m6-quality m6-test m1-smoke m3-smoke m4-smoke m5-smoke m6-data m6-eval m6-smoke


m7-quality:
	python -m ruff check .

m7-test:
	python -m pytest -q

m7-run:
	python scripts/m7_analyze.py

m7-smoke:
	python scripts/m7_smoke.py

m7-all: m7-quality m7-test m1-smoke m3-smoke m4-smoke m5-smoke m6-smoke m7-run m7-smoke


m8-quality:
	python -m ruff check .

m8-test:
	python -m pytest -q

m8-freeze:
	python scripts/m8_freeze.py

m8-smoke:
	python scripts/m8_smoke.py

m8-api-smoke:
	python scripts/m8_api_smoke.py

m8-serve:
	SPAM_FILTER_ROOT=$(CURDIR) python -m uvicorn spam_filter.api:app --host 127.0.0.1 --port 8000

m8-all: m8-quality m8-test m6-smoke m7-smoke m8-freeze m8-smoke m8-api-smoke

m9-quality:
	python -m ruff check .

m9-test:
	python -m pytest -q

m9-benchmark:
	python scripts/m9_benchmark.py

m9-release:
	python scripts/m9_release.py

m9-smoke:
	python scripts/m9_smoke.py

m9-api-smoke:
	python scripts/m9_api_smoke.py

m9-serve:
	SPAM_FILTER_ROOT=$(CURDIR) python -m uvicorn spam_filter.api:app --host 127.0.0.1 --port 8000

m9-all: m9-quality m9-test m6-smoke m7-smoke m8-smoke m9-benchmark m9-release m9-smoke m9-api-smoke
