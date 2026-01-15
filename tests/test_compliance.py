import contextlib
import io
import os
import struct
import unittest

from rv32sim import RV32Sim, HaltException, TrapException


SUITE_RISCV_TESTS = "riscv-tests"
SUITE_ARCH_TESTS = "riscv-arch-test"
SUITE_TORTURE = "riscv-torture"

RISCV_TESTS_DIR = "tests/elfs/riscv-tests"
ARCH_TESTS_DIR = "tests/elfs/riscv-arch-test"
TORTURE_ELF = "tests/elfs/riscv-torture/test.elf"

TOHOST_RISCV_TESTS = 0x80001000
TOHOST_ARCH_TESTS = 0x80003000
TOHOST_TORTURE = 0x80001000

MAX_STEPS = int(os.getenv("RV32SIM_MAX_STEPS", "2000000"))
STACK_SLACK = 0x10000
PAGE_SIZE = 0x1000


def _align_down(value, align):
    return value & ~(align - 1)


def _align_up(value, align):
    return (value + align - 1) & ~(align - 1)


def _read_elf(path):
    with open(path, "rb") as handle:
        elf = handle.read()

    if len(elf) < 52 or elf[:4] != b"\x7fELF":
        raise ValueError(f"Invalid ELF file: {path}")
    if elf[4] != 1 or elf[5] != 1:
        raise ValueError(f"Unsupported ELF class or endianness: {path}")
    return elf


def _read_cstr(blob, offset):
    if blob is None or offset >= len(blob):
        return None
    end = blob.find(b"\x00", offset)
    if end == -1:
        end = len(blob)
    return blob[offset:end].decode("ascii", errors="replace")


def _iter_sections(elf):
    e_shoff = struct.unpack_from("<I", elf, 32)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 46)
    if e_shoff == 0 or e_shnum == 0 or e_shentsize < 40:
        return []
    if e_shoff + e_shentsize * e_shnum > len(elf):
        raise ValueError("Section headers extend beyond EOF")

    shstr = None
    if e_shstrndx < e_shnum:
        shstr_off = e_shoff + e_shstrndx * e_shentsize
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from(
            "<IIIIII", elf, shstr_off
        )
        if sh_offset + sh_size <= len(elf):
            shstr = elf[sh_offset:sh_offset + sh_size]

    sections = []
    for idx in range(e_shnum):
        off = e_shoff + idx * e_shentsize
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = (
            struct.unpack_from("<IIIIIIIIII", elf, off)
        )
        sections.append({
            "name": _read_cstr(shstr, sh_name),
            "type": sh_type,
            "addr": sh_addr,
            "offset": sh_offset,
            "size": sh_size,
            "link": sh_link,
            "entsize": sh_entsize,
        })
    return sections


def _find_tohost_addr(elf):
    sections = _iter_sections(elf)
    for section in sections:
        if section["name"] == ".tohost":
            return section["addr"]

    for section in sections:
        if section["type"] != 2 or section["entsize"] < 16:
            continue
        link = section["link"]
        if link >= len(sections):
            continue
        strtab = sections[link]
        if strtab["offset"] + strtab["size"] > len(elf):
            continue
        names = elf[strtab["offset"]:strtab["offset"] + strtab["size"]]
        for ent_off in range(section["offset"], section["offset"] + section["size"], section["entsize"]):
            if ent_off + 16 > len(elf):
                break
            st_name, st_value, st_size, st_info, st_other, st_shndx = struct.unpack_from(
                "<IIIBBH", elf, ent_off
            )
            name = _read_cstr(names, st_name)
            if name == "tohost":
                return st_value
    return None


def _load_range(elf, path):
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


def _suite_enabled(name):
    suite = os.getenv("RV32SIM_SUITE")
    return not suite or suite == name


def _chunk_paths(paths):
    chunk_count = int(os.getenv("RV32SIM_CHUNK_COUNT", "1"))
    chunk_index = int(os.getenv("RV32SIM_CHUNK_INDEX", "0"))
    if chunk_count <= 1:
        return paths
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("Invalid RV32SIM_CHUNK_INDEX/RV32SIM_CHUNK_COUNT")
    return [path for idx, path in enumerate(paths) if idx % chunk_count == chunk_index]


def run_elf(path, tohost_addr, max_steps=MAX_STEPS):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    elf = _read_elf(path)
    mem_start, mem_end = _load_range(elf, path)
    elf_tohost = _find_tohost_addr(elf)
    if elf_tohost is not None:
        tohost_addr = elf_tohost
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
                # Emulate trap entry; fall back to a stub if mtvec isn't mapped.
                trap_pc = sim._raise_trap(exc.cause, exc.tval)
                if trap_pc == 0 or sim.is_hardware_region(trap_pc):
                    instr = sim.last_instr if sim.last_instr is not None else 0
                    instr_size = 2 if (instr & 0x3) != 0x3 else 4
                    mstatus = sim._csr_read(0x300)
                    mpp = (mstatus >> 11) & 0x3
                    sim.priv = mpp
                    sim._csr_write(0x300, mstatus & ~0x1800)
                    mepc = sim.csrs.get(0x341, sim.pc) & 0xffffffff
                    sim.pc = (mepc + instr_size) & 0xffffffff
                else:
                    sim.pc = trap_pc
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
        if not _suite_enabled(SUITE_RISCV_TESTS):
            self.skipTest("suite filtered")
        paths = _chunk_paths(sorted(
            os.path.join(RISCV_TESTS_DIR, name)
            for name in os.listdir(RISCV_TESTS_DIR)
            if os.path.isfile(os.path.join(RISCV_TESTS_DIR, name))
        ))
        if not paths:
            self.skipTest("no ELF files in this chunk")
        for path in paths:
            with self.subTest(path=path):
                exc = run_elf(path, TOHOST_RISCV_TESTS)
                self._assert_pass(exc, path)

    def test_riscv_arch_tests(self):
        if not _suite_enabled(SUITE_ARCH_TESTS):
            self.skipTest("suite filtered")
        paths = _chunk_paths(sorted(
            os.path.join(ARCH_TESTS_DIR, name)
            for name in os.listdir(ARCH_TESTS_DIR)
            if os.path.isfile(os.path.join(ARCH_TESTS_DIR, name))
        ))
        if not paths:
            self.skipTest("no ELF files in this chunk")
        for path in paths:
            with self.subTest(path=path):
                exc = run_elf(path, TOHOST_ARCH_TESTS)
                self._assert_pass(exc, path)

    def test_riscv_torture(self):
        if not _suite_enabled(SUITE_TORTURE):
            self.skipTest("suite filtered")
        exc = run_elf(TORTURE_ELF, TOHOST_TORTURE)
        self._assert_pass(exc, TORTURE_ELF)
