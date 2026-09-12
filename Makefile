# Thin wrapper over the cross-platform task runner, for Linux and CI.
# On Windows, run the same tasks directly: python -m ips.tasks <task>
PY ?= python

.PHONY: generate sample dbt docs segment test all

generate:
	$(PY) -m ips.tasks generate

sample:
	$(PY) -m ips.tasks generate --sample

dbt:
	$(PY) -m ips.tasks dbt

docs:
	$(PY) -m ips.tasks docs

segment:
	$(PY) -m ips.tasks segment

test:
	$(PY) -m ips.tasks test

all:
	$(PY) -m ips.tasks all
