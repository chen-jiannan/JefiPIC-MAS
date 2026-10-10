.PHONY: test cpu-repro env gpu-quick clean
test:
	pytest -q
cpu-repro:
	python scripts/run_cpu_repro.py
env:
	python scripts/check_environment.py
gpu-quick:
	python scripts/run_gpu_quick.py
clean:
	rm -rf reproduced_results article_cases_output .pytest_cache
