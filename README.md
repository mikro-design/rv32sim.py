# rv32sim: A Python-based RV32I Simulator with GDB Stub

`rv32sim` is a lightweight, extensible RISC-V 32-bit simulator written in Python. It implements the RV32I base integer instruction set (plus M and partial C extensions) and features a built-in GDB remote stub, allowing you to debug bare-metal RISC-V programs using standard tools.

It is designed for educational purposes, emulator development, and testing basic bare-metal software without the overhead of heavy simulation environments like QEMU.

## Features

*   **RV32IMC Support:** Implements the base RV32I instruction set, standard multiplication/division (M) extension, and common compressed (C) instructions.
*   **GDB Remote Stub:** Acts as a GDB server (RSP protocol), allowing connection from `gdb-multiarch` or `riscv32-unknown-elf-gdb`.
    *   Support for breakpoints, stepping, register/memory inspection, and modification.
*   **MMIO & Hardware Stubbing:**
    *   Define hardware peripherals and registers via JSON configuration.
    *   **Detect Mode:** Automatically record unknown MMIO accesses to generate stub templates.
    *   **Assisted Mode:** Interactive prompts for discovering and handling new hardware accesses.
*   **Assertion Engine:**
    *   Validate MMIO writes against expected values or bitmasks defined in JSON.
    *   Useful for regression testing of driver code.
*   **RTT (Real Time Transfer):** Basic support for SEGGER RTT-like communication.
*   **Extensible Memory Map:** Configurable memory regions (Flash, SRAM) via CLI or JSON.

## Installation

No special installation is required other than Python 3.7+.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/rv32sim.git
    cd rv32sim
    ```

2.  **Dependencies:**
    Standard Python libraries are used. `svd_parser` is included or required if you use SVD features.

## Quick Start

### 1. Compiling a Program
You need a RISC-V toolchain. See `examples/` for a sample `Makefile`.

```bash
cd examples
make
cd ..
```

### 2. Running Tests
You can run the automated test suite to verify everything is working (requires `pytest`):

```bash
python3 -m pip install pytest
make test
```

If you have a RISC-V toolchain installed, you can also run the example-driven integration tests:

```bash
make test-examples
```

### 2. Running the Simulator
Run the simulator (starts empty, waiting for GDB):

```bash
python3 rv32sim.py
```
The simulator will start and wait for a GDB connection on port 3333.

### 3. Connecting GDB
In a separate terminal:

```bash
riscv32-unknown-elf-gdb examples/debug_example.elf
(gdb) target remote :3333
(gdb) load
(gdb) break main
(gdb) continue
```

## detailed Usage

```
python3 rv32sim.py [program.elf] [OPTIONS]
```

### Options
*   `--port=PORT`: GDB server port (default: 3333).
*   `--stub=FILE`: Load hardware register stubs from a JSON file.
*   `--detect`: Enable hardware access detection. Captures unknown MMIO R/W and saves them to `hw_stubs.json` on exit.
*   `--assisted`: Interactive mode that pauses on unknown MMIO and suggests stubs.
*   `--mem-region=START:SIZE[:NAME]`: Add custom memory regions (e.g., `--mem-region=0x80000000:0x10000:dram`).
*   `--assert=FILE`: Load MMIO assertions from a JSON file.
*   `--assert-writes`: Check writes against assertions (default is to only track).
*   `--svd=FILE`: Load a CMSIS-SVD file to provide human-readable register names during simulation/assertion.
*   `--uart-input=STRING`: Pre-load UART RX buffer with a string (supports \n).

## Configuration Files

### Hardware Stubs (`hw_stubs.json`)
Used to mock hardware registers so firmware doesn't hang waiting for status bits.

```json
{
  "hw_stubs": {
    "0x40000000": 1,        // Return 1 when reading 0x40000000
    "0x50000004": {
      "value": "0xDEADBEEF",
      "comment": "Status register"
    }
  }
}
```

### Assertions (`assertions.json`)
Used to verify that the software interacts with hardware correctly.

```json
{
  "assertions": {
    "0x40000000": {
      "register": "UART_TX",
      "write": {
        "mask": "0xFF",     // Only check lower 8 bits
        "value": "0x41"     // Expect 'A' to be written
      }
    }
  }
}
```

## Architecture

*   **`rv32sim.py`**: Entry point. Manages the simulation loop, GDB server, and CLI arguments.
*   **`cpu_core.py`**: Implements the CPU logic (fetch, decode, execute) and instruction handlers.
*   **`memory_map.py`**: Handles memory regions (RAM/ROM) and routing.
*   **`mmio.py`**: Manages Memory Mapped I/O, hooks, and stubs.
*   **`gdb_server.py`**: Implements the GDB Remote Serial Protocol (RSP).
*   **`assertion_manager.py`**: Handles checking of memory accesses against defined rules.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repo.
2.  Create your feature branch (`git checkout -b my-new-feature`).
3.  Commit your changes (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin my-new-feature`).
5.  Create a new Pull Request.

## License

[BSD 3-Clause License](LICENSE)
