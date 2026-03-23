.PHONY: build smoke perft uci

build:
	./scripts/build.sh

perft: build
	./engine/cortex perft 5

uci: build
	printf 'uci\nisready\nquit\n' | ./engine/cortex

# Vollpipeline inkl. Trainer (legt ggf. trainer/.venv an, pip install)
smoke:
	./scripts/smoke.sh
