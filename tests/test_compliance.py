import os
import unittest

from rv32sim import RV32Sim, HaltException, TrapException


ELF_CASES = [
    ("tests/elfs/riscv-tests/rv32ui-p-add", 0x80001000, "exit", 0),
    ("tests/elfs/riscv-arch-test/I-add-01.elf", 0x80003000, "tohost", 1),
    ("tests/elfs/riscv-torture/test.elf", 0x80001000, "tohost", 1),
]

MEM_START = 0x80000000
MEM_END = 0x80010000
MAX_STEPS = 2_000_000


def run_elf(path, tohost_addr, max_steps=MAX_STEPS):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(MEM_START, MEM_END, "ram")
    sim.configure_tohost(tohost_addr, size=8)
    sim.load_elf(path)
    sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff

    steps = 0
    while steps < max_steps:
        try:
            sim.execute()
        except TrapException as exc:
            # Emulate trap entry so compliance tests can run their handlers.
            sim.pc = sim._raise_trap(exc.cause, exc.tval)
        except HaltException as exc:
            return exc
        steps += 1

    raise RuntimeError(f"Timeout after {max_steps} steps")


class ComplianceElfTests(unittest.TestCase):
    def test_compliance_elfs(self):
        for path, tohost_addr, expected_reason, expected_code in ELF_CASES:
            with self.subTest(path=path):
                exc = run_elf(path, tohost_addr)
                self.assertEqual(exc.reason, expected_reason)
                self.assertEqual(exc.code, expected_code)
