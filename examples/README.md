# rv32sim Examples

This directory contains example programs to demonstrate the features of `rv32sim`.

## Requirements

To build the examples, you need a RISC-V GCC toolchain (e.g., `riscv32-unknown-elf-gcc`).
Ubuntu/Debian: `sudo apt install gcc-riscv64-unknown-elf` (supports 32-bit via flags).

## Building

Run `make` to build all examples:

```bash
make
```

## Example 1: Debugging (Basic)

`debug_example.c` is a simple program that calculates Fibonacci numbers.

1.  **Start:** `python3 ../rv32sim.py` (Starts empty, waiting for GDB)
2.  **Connect:**
    ```bash
    riscv32-unknown-elf-gdb debug_example.elf
    (gdb) target remote :3333
    (gdb) load
    (gdb) break main
    (gdb) continue
    ```
3.  **Debug:** Step through code, inspect variables.

## Example 2: Assertions (MMIO Validation)

`assertion_example.c` writes to specific memory addresses (UART peripheral). `assertions.json` defines expected values.

1.  **Run:**
    ```bash
    python3 ../rv32sim.py --assert assertions.json --assert-writes
    ```
2.  **Connect (GDB):**
    ```bash
    riscv32-unknown-elf-gdb assertion_example.elf
    (gdb) target remote :3333
    (gdb) load
    (gdb) continue
    ```
3.  **Verify:** The simulator validates each write against the JSON rules. Try changing `assertion_example.c` to write a wrong value (e.g., `0x42` instead of `0x41`) to see it fail.

## Example 3: RTT (Real Time Transfer)

`rtt_example.c` demonstrates bi-directional communication using SEGGER RTT protocol without a UART.

1.  **Start Simulator:**
    ```bash
    # OpenOCD mode enables RTT on port 4001 (data) and 4444 (cmd)
    python3 ../rv32sim.py --rtt-openocd
    ```
2.  **Connect RTT Client:**
    Open a new terminal and connect via Telnet:
    ```bash
    telnet localhost 4001
    ```
3.  **Load & Run (GDB):**
    ```bash
    riscv32-unknown-elf-gdb rtt_example.elf
    (gdb) target remote :3333
    (gdb) load
    (gdb) continue
    ```
4.  **Interact:**
    *   You should see "RTT Initialized..." and "Tick..." messages.
    *   Type characters in the telnet window. The simulator will read them and echo them back ("Echo: X").

## Example 4: Matrix Multiplication (Compute & Syscalls)

`matrix_example.c` performs a 16x16 matrix multiplication, verifies the result, and prints to stdout using system calls.

1.  **Run:**
    ```bash
    python3 ../rv32sim.py matrix_example.elf
    ```
2.  **Output:**
    The simulator traps the `ecall` (write) and prints directly to your terminal:
    ```
    Starting Matrix Multiplication (16x16)...
    Computation done. Checking corner values:
    C[0][0] = 1632
    ...
    SUCCESS: C[0][0] matches expected value.
    ```
