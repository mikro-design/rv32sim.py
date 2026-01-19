SHELL := /bin/bash

GDB ?= riscv32-unknown-elf-gdb
SIM ?= python3 ../rv32sim.py
PORT ?= 3333
TIMEOUT ?= 10

RISCV_TESTS_DIR := elfs/riscv-tests
ARCH_TESTS_DIR := elfs/riscv-arch-test

RISCV_TESTS := $(wildcard $(RISCV_TESTS_DIR)/*)
ARCH_TESTS := $(wildcard $(ARCH_TESTS_DIR)/*.elf)

.PHONY: all riscv-tests riscv-arch-test

all: riscv-tests riscv-arch-test

riscv-tests:
	@$(MAKE) --no-print-directory run-suite SUITE="$(RISCV_TESTS)"

riscv-arch-test:
	@$(MAKE) --no-print-directory run-suite SUITE="$(ARCH_TESTS)"

run-suite:
	@set -euo pipefail; \
	fails=0; total=0; \
	for elf in $(SUITE); do \
		if [ ! -f "$$elf" ]; then continue; fi; \
		total=$$((total+1)); \
		printf "[TEST] %s\n" "$$elf"; \
		$(SIM) --port=$(PORT) >/tmp/rv32sim_$$.log 2>&1 & \
		sim_pid=$$!; \
		sleep 0.2; \
		$(GDB) --batch -ex "target remote :$(PORT)" -ex "load" -ex "continue" "$$elf" >/tmp/gdb_$$.log 2>&1; \
		status=$$?; \
		kill $$sim_pid >/dev/null 2>&1 || true; \
		wait $$sim_pid >/dev/null 2>&1 || true; \
		if [ $$status -ne 0 ]; then \
			printf "[FAIL] %s (gdb exit %s)\n" "$$elf" "$$status"; \
			fails=$$((fails+1)); \
		else \
			printf "[PASS] %s\n" "$$elf"; \
		fi; \
	done; \
	printf "\n[RESULT] %s/%s passed\n" $$((total-fails)) $$total; \
	if [ $$fails -ne 0 ]; then exit 1; fi
