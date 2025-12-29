import contextlib
import io
import os
import struct
import unittest

from rv32sim import RV32Sim, HaltException, TrapException


RISCV_TESTS_DIR = "tests/elfs/riscv-tests"
ARCH_TESTS_DIR = "tests/elfs/riscv-arch-test"
TORTURE_ELF = "tests/elfs/riscv-torture/test.elf"

TOHOST_RISCV_TESTS = 0x80001000
TOHOST_ARCH_TESTS = 0x80003000
TOHOST_TORTURE = 0x80001000

MAX_STEPS = 2_000_000
STACK_SLACK = 0x10000
PAGE_SIZE = 0x1000


def _align_down(value, align):
    return value & ~(align - 1)


def _align_up(value, align):
    return (value + align - 1) & ~(align - 1)


def _load_range(path):
    with open(path, "rb") as handle:
        elf = handle.read()

    if len(elf) < 52 or elf[:4] != b"\x7fELF":
        raise ValueError(f"Invalid ELF file: {path}")
    if elf[4] != 1 or elf[5] != 1:
        raise ValueError(f"Unsupported ELF class or endianness: {path}")

    e_phoff = struct.unpack_from("<I", elf, 28)[0]
    e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 42)
    if e_phentsize < 32:
        raise ValueError(f"Invalid program header size: {path}")

    min_addr = None
    max_addr = None
    for idx in range(e_phnum):
        off = e_phoff + idx * e_phentsize
        if off + e_phentsize > len(elf):
            raise ValueError(f"Program headers extend beyond EOF: {path}")
        p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz = struct.unpack_from(
            "<IIIIIIII", elf, off
        )[:6]
        if p_type != 1:
            continue
        seg_start = min(p_vaddr, p_paddr)
        seg_end = max(p_vaddr + p_memsz, p_paddr + p_memsz)
        min_addr = seg_start if min_addr is None else min(min_addr, seg_start)
        max_addr = seg_end if max_addr is None else max(max_addr, seg_end)

    if min_addr is None or max_addr is None:
        raise ValueError(f"No loadable segments: {path}")

    start = _align_down(min_addr, PAGE_SIZE)
    end = _align_up(max_addr, PAGE_SIZE) + STACK_SLACK
    return start, end


def run_elf(path, tohost_addr, max_steps=MAX_STEPS):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    mem_start, mem_end = _load_range(path)
    if tohost_addr < mem_start:
        mem_start = _align_down(tohost_addr, PAGE_SIZE)
    if tohost_addr + 8 > mem_end:
        mem_end = _align_up(tohost_addr + 8, PAGE_SIZE) + STACK_SLACK

    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(mem_start, mem_end, "ram")
    sim.configure_tohost(tohost_addr, size=8)
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent):
        sim.load_elf(path)
        sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff

        steps = 0
        while steps < max_steps:
            try:
                sim.execute()
            except TrapException as exc:
                # Emulate trap entry so compliance tests can run their handlers.
                sim.pc = sim._raise_trap(exc.cause, exc.tval)
            except ValueError as exc:
                msg = str(exc)
                if "Illegal" in msg or "Unsupported" in msg:
                    sim.pc = sim._raise_trap(2, sim.last_instr or 0)
                else:
                    raise
            except HaltException as exc:
                return exc
            steps += 1

    raise RuntimeError(f"Timeout after {max_steps} steps")


class ComplianceElfTests(unittest.TestCase):
    def _assert_pass(self, exc, path):
        if exc.reason == "exit":
            self.assertEqual(exc.code, 0, msg=f"{path} exit code {exc.code}")
            return
        if exc.reason == "tohost":
            self.assertEqual(exc.code, 1, msg=f"{path} tohost value {exc.code}")
            return
        self.fail(f"{path} halted with {exc.reason} {exc.code}")

    def test_riscv_tests(self):
        for path in sorted(
            os.path.join(RISCV_TESTS_DIR, name)
            for name in os.listdir(RISCV_TESTS_DIR)
            if os.path.isfile(os.path.join(RISCV_TESTS_DIR, name))
        ):
            with self.subTest(path=path):
                exc = run_elf(path, TOHOST_RISCV_TESTS)
                self._assert_pass(exc, path)

    def test_riscv_arch_tests(self):
        for path in sorted(
            os.path.join(ARCH_TESTS_DIR, name)
            for name in os.listdir(ARCH_TESTS_DIR)
            if os.path.isfile(os.path.join(ARCH_TESTS_DIR, name))
        ):
            with self.subTest(path=path):
                exc = run_elf(path, TOHOST_ARCH_TESTS)
                self._assert_pass(exc, path)

    def test_riscv_torture(self):
        exc = run_elf(TORTURE_ELF, TOHOST_TORTURE)
        self._assert_pass(exc, TORTURE_ELF)
