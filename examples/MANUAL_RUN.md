# Manual Execution Guide

This document provides the exact command lines to run each example "one by one", demonstrating both the simulator's standalone mode and the GDB Remote Load workflow.

**Prerequisite:** Open two terminals.
**GDB Path:** `riscv32-unknown-elf-gdb`

---

## 1. Debug Example (Classic GDB Workflow)

**Terminal 1 (Simulator):**
Start the simulator empty, waiting for a connection.
```bash
python3 rv32sim.py --port 3333
```

**Terminal 2 (GDB):**
Connect, upload code, and debug.
```bash
riscv32-unknown-elf-gdb examples/debug_example.elf \
    -ex "target remote :3333" \
    -ex "load" \
    -ex "break main" \
    -ex "continue"
```

---

## 2. Assertions Example (Verification Workflow)

The simulator needs to know the assertions *before* the code runs to validate the MMIO writes.

Create `examples/assertions.json` with the following contents:

```json
{
  "assertions": {
    "0x40000000": {
      "register": "UART_DATA",
      "write": { "value": "0x41", "mask": "0xFF" }
    },
    "0x40000004": {
      "register": "UART_CTRL",
      "read": { "value": "0x0" },
      "write": { "value": "0x1", "mask": "0x1" }
    }
  }
}
```

**Terminal 1 (Simulator):**
```bash
python3 rv32sim.py --port 3333 --assert examples/assertions.json --assert-writes
```

**Terminal 2 (GDB):**
```bash
riscv32-unknown-elf-gdb examples/assertion_example.elf \
    -ex "target remote :3333" \
    -ex "load" \
    -ex "continue"
```
*Note: If the code writes a bad value to 0x40000000, Terminal 1 will show an error.*

---

## 3. UART Example (Interactive Input)

**Terminal 1 (Simulator):**
We can pre-load the UART buffer with input string.
```bash
python3 rv32sim.py --port 3333 --uart-input="HelloGDB\nq"
```

**Terminal 2 (GDB):**
```bash
riscv32-unknown-elf-gdb examples/uart_polling.elf \
    -ex "target remote :3333" \
    -ex "load" \
    -ex "continue"
```
*Terminal 1 will display the echoed characters.*

---

## 4. RTT Example (Advanced IO)

This requires **three** terminals: Simulator, RTT Client, and GDB.

**Terminal 1 (Simulator):**
Enable the RTT server port.
```bash
python3 rv32sim.py --port 3333 --rtt-port 4444
```

**Terminal 2 (Telnet - Run this BEFORE GDB 'continue'):**
```bash
telnet localhost 4444
```

**Terminal 3 (GDB):**
```bash
riscv32-unknown-elf-gdb examples/rtt_example.elf \
    -ex "target remote :3333" \
    -ex "load" \
    -ex "continue"
```
*Type in Terminal 2 to see echo.*

---

## 5. Matrix Example (Compute & Syscalls)

This is best run in standalone mode because it uses Semihosting (Syscalls) to print to the simulator console.

**Terminal 1:**
```bash
python3 rv32sim.py examples/matrix_example.elf
```

**Alternative (GDB Way):**
You *can* load it via GDB, but the output will appear in Terminal 1.

**Terminal 1:**
```bash
python3 rv32sim.py --port 3333
```

**Terminal 2:**
```bash
riscv32-unknown-elf-gdb examples/matrix_example.elf \
    -ex "target remote :3333" \
    -ex "load" \
    -ex "continue"
```

---

## 6. CSR Example (Architecture Tests)

**Terminal 1:**
```bash
python3 rv32sim.py examples/csr_example.elf
```
