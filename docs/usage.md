# Usage guide

rv32sim can be used in two ways:

1) As a CLI that starts a GDB remote server.
2) As a Python module you drive directly.

## CLI (GDB remote)

Start the simulator and load an ELF:

```bash
python rv32sim.py tests/elfs/riscv-tests/rv32ui-p-add
```

This starts a GDB server on port 3333 and waits for a debugger connection. In
another terminal:

```bash
gdb-multiarch tests/elfs/riscv-tests/rv32ui-p-add
```

```gdb
(gdb) target remote :3333
(gdb) continue
```

Common GDB commands:

- stepi: execute one instruction
- info registers: show register state
- x/16wx 0x20000000: inspect memory

If you do not have gdb-multiarch, use a RISC-V capable gdb build and point it
at the ELF.

## Stub configuration

Many ELFs expect MMIO registers. Use a stub config to define memory regions and
MMIO behavior:

```bash
python rv32sim.py tests/elfs/riscv-arch-test/I-add-01.elf --stub examples/hw_stubs.json
```

See docs/hw-stubs.md for the full JSON format.

## Detect and assisted modes

Use detect mode to discover unknown MMIO addresses and generate a stub
template on exit:

```bash
python rv32sim.py tests/elfs/riscv-torture/test.elf --detect
```

Assisted mode is the same but prints hints when new MMIO addresses are seen:

```bash
python rv32sim.py tests/elfs/riscv-torture/test.elf --assisted
```

Stop the simulator with Ctrl-C to write the generated stub JSON.

## Using as a Python module

You can embed the simulator directly:

```python
from rv32sim import RV32Sim, HaltException

sim = RV32Sim()
sim.load_elf("tests/elfs/riscv-tests/rv32ui-p-add")
sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff

try:
    while True:
        sim.execute()
except HaltException as exc:
    print(f"Halt: {exc.reason} code={exc.code}")
```

If your program does not write to tohost, it may run indefinitely. Add a step
limit if needed.

## Running the unit tests

```bash
python3 -m unittest discover -s tests
```
