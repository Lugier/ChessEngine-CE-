.PHONY: build verify smoke perft uci bench sprt

build:
	./scripts/build.sh

# Strikte Checks: Perft 1–5, UCI, Dataprep, Python-Syntax, optional Trainer
verify:
	./scripts/verify.sh

# Alias zu verify (Trainer läuft standardmäßig; ohne: SKIP_TRAINER=1 make smoke)
smoke:
	./scripts/smoke.sh

perft: build
	./engine/cortex perft 5

uci: build
	printf 'uci\nisready\nquit\n' | ./engine/cortex

bench: build
	./scripts/bench_strength.sh

sprt: build
	./scripts/sprt.sh
