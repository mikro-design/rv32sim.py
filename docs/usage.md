# rv32sim Usage Guide

This document provides detailed information on how to use `rv32sim` effectively.

## Command Line Interface

The general syntax is:
```bash
python rv32sim.py [program.elf] [OPTIONS]
```

If no ELF file is provided, the simulator starts with empty memory (useful if you plan to `load` via GDB).

### GDB Server Options

*   **`--port=PORT`**: Sets the TCP port for the GDB server. Default is `3333`.
    *   Example: `--port=1234`

### Hardware Stubbing & Detection

These options help you run firmware that expects specific hardware peripherals.

*   **`--stub=FILE`**: Loads a JSON file defining values for specific memory addresses. This is crucial for bypassing hardware initialization loops (e.g., waiting for a PLL lock bit).
*   **`--detect`**: Runs the simulator in "detection mode". Any access to memory not defined in `memory_regions` is logged. On exit, a `hw_stubs.json` file is generated containing these accesses, which you can then edit and use as a stub file.
*   **`--assisted`**: An interactive version of detect mode. When an unknown MMIO access occurs, the simulator pauses and prints details (PC, address, access type).

### Assertions

Used for verifying driver behavior.

*   **`--assert=FILE`**: Loads an assertion configuration file.
*   **`--assert-writes`**: By default, assertions might just be monitored. This flag enforces them, stopping or reporting errors on writes.
*   **`--assert-assist`**: An interactive mode to help build assertion files. It prompts you when MMIO accesses occur, allowing you to define rules on the fly.
*   **`--assert-verbose`**: Prints detailed information about fields if an SVD file is loaded.
*   **`--svd=FILE`**: Loads a CMSIS-SVD (System View Description) file. This allows the simulator to display human-readable register and field names instead of raw addresses.

### Memory Configuration

*   **`--mem-region=START:SIZE[:NAME]`**: Adds a RAM region.
    *   `START`: Hex or decimal start address.
    *   `SIZE`: Size in bytes.
    *   `NAME`: Optional name (e.g., `sram`, `dram`).
    *   Example: `--mem-region=0x20000000:0x8000:sram` (32KB SRAM at 0x20000000).

## GDB Usage Tips

Once connected to the simulator via `target remote :3333`:

*   **`load`**: Uploads the ELF file sections to the simulator's memory. This is often required even if you provided the ELF on the command line, to ensure the CPU state is reset and memory is fresh.
*   **`monitor reset_counter`**: Resets the internal cycle and instruction counters.
*   **`monitor show_stats`**: Displays execution statistics (cycle count, instruction count).
*   **`monitor run_steps N`**: Executes N instructions and stops.
*   **`monitor load_elf PATH`**: Loads a new ELF file at runtime.

## Stubbing Workflow

1.  **Initial Run:** Run your firmware with `--detect`.
    ```bash
    python rv32sim.py firmware.elf --detect
    ```
2.  **Generate Stubs:** Interact with the firmware until it crashes or you exit. A `hw_stubs.json` (or similar name based on ELF) will be created.
3.  **Refine Stubs:** Edit the generated JSON.
    *   Change values from `0` to whatever the firmware expects (e.g., `0x1` for a "ready" bit).
    *   Add comments.
4.  **Run with Stubs:**
    ```bash
    python rv32sim.py firmware.elf --stub hw_stubs.json
    ```

## Assertion Workflow

1.  **Assist Mode:** Run with `--assert-assist` and optionally `--svd` for better context.
    ```bash
    python rv32sim.py firmware.elf --assert-assist --svd device.svd
    ```
2.  **Define Rules:** As the firmware runs, the CLI will prompt you on MMIO accesses. You can specify:
    *   Expected values.
    *   Bitmasks (which bits matter).
    *   Sequences (for reading a FIFO or status change).
3.  **Save:** On exit, an assertion JSON file is saved.
4.  **Verify:** Run future regressions with:
    ```bash
    python rv32sim.py firmware.elf --assert assertions.json --assert-writes
    ```

## Python API Usage

You can embed the simulator directly in Python scripts:

```python
from rv32sim import RV32Sim, HaltException

sim = RV32Sim()
sim.load_elf("tests/elfs/riscv-tests/rv32ui-p-add")
# Initialize stack pointer if not set by _start
sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff

try:
    while True:
        sim.execute()
except HaltException as exc:
    print(f"Halt: {exc.reason} code={exc.code}")
```
