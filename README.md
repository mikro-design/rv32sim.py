# rv32sim

RV32IMC simulator with a GDB remote stub. It loads ELF32 little-endian RISC-V
binaries, supports MMIO stubs via JSON, and can detect unknown hardware
accesses.

## Quick start

python rv32sim.py tests/elfs/riscv-arch-test/I-add-01.elf --stub examples/hw_stubs.json

## Features

- ELF32 loader for RISC-V
- RV32I core with M and partial C (compressed) support
- GDB remote stub (default port 3333; use --port=PORT)
- Hardware register stubbing and function stubs via JSON
- Detect and assisted modes to generate stub templates

## Layout

- rv32sim.py: simulator and GDB stub
- examples/hw_stubs.json: sample stub configuration
- tests/test_rv32sim.py: unit tests
- tests/elfs/: ELF binaries from RISC-V test suites
- third_party/: upstream licenses for included test binaries

## Usage

python rv32sim.py <program.elf> [--port=PORT] [--stub=FILE] [--detect|--assisted]

The CLI starts a GDB server and waits for a debugger connection. Use GDB to run
the program (see examples below or docs/usage.md).

When --detect is enabled, the simulator records unknown MMIO accesses and
writes a stub template on exit. Assisted mode prints hints as new MMIO
addresses are discovered.

## Examples

- Run a small ISA test:
  python rv32sim.py tests/elfs/riscv-tests/rv32ui-p-add
- Run with a stub config:
  python rv32sim.py tests/elfs/riscv-arch-test/I-add-01.elf --stub examples/hw_stubs.json
- Generate a stub template from unknown MMIO:
  python rv32sim.py tests/elfs/riscv-torture/test.elf --detect
- Attach GDB (in another terminal):
  gdb-multiarch tests/elfs/riscv-tests/rv32ui-p-add
  (gdb) target remote :3333
  (gdb) continue

## Testing

python3 -m unittest discover -s tests

The compliance harness runs all ELF binaries under `tests/elfs/`.
CI splits the suite with `RV32SIM_SUITE`, `RV32SIM_CHUNK_COUNT`, and
`RV32SIM_CHUNK_INDEX` (optionally `RV32SIM_MAX_STEPS`).

## Notes

- Default memory regions are flash 0x00000000-0x00020000 and SRAM at
  0x20000000 with size set by memory_size (default 16 MB). You can override
  with memory_regions in the stub JSON.
- Some compressed instructions are not implemented and will raise an error
  if encountered.

## More docs

- docs/hw-stubs.md
- docs/test-elfs.md
- docs/usage.md
