import json
import runpy
import struct
import sys
from pathlib import Path

import pytest

from rv32sim import RV32Sim, TrapException


class _BadMemIf:
    name = "bad"

    def access(self, addr, size, is_write=False):
        raise RuntimeError("boom")

    def stats(self):
        return {"accesses": 2, "hits": 1, "misses": 1}


def _build_elf(phdr, entry=0, phoff=52, phentsize=32, phnum=1, e_machine=243, e_ident_class=1, e_ident_data=1):
    e_ident = bytearray(16)
    e_ident[0:4] = b"\x7fELF"
    e_ident[4] = e_ident_class
    e_ident[5] = e_ident_data
    e_ident[6] = 1
    header = struct.pack("<16sHHIII", bytes(e_ident), 2, e_machine, 1, entry, phoff)
    header += struct.pack("<IIHHHHHH", 0, 0, 52, phentsize, phnum, 0, 0, 0)
    data = header
    if len(data) < phoff:
        data += b"\x00" * (phoff - len(data))
    data += phdr
    return data


def test_branch_predictor_paths():
    sim = RV32Sim()
    sim._configure_branch_predictor(
        {"enabled": True, "entries": 1, "store_forward": False, "alias_word": True}
    )
    assert sim._bp_lookup(0x100) is None
    assert sim._bp_should_store(0x100, 0x200) is False
    assert sim._bp_should_store(0x200, 0x100) is True
    sim._bp_insert(0x100, 0x80)
    assert sim._bp_lookup(0x100) == 0
    assert sim._branch_penalty(0x100, 0x80, True) == 0
    assert sim._branch_penalty(0x100, 0x84, True) == sim.timing["mispredict_penalty"]
    assert sim._branch_penalty(0x100, 0x80, False) == 0

    sim._configure_branch_predictor({"enabled": False, "entries": 0})
    assert sim._branch_penalty(0x100, 0x80, True) == sim.timing["branch_penalty"]


def test_memif_access_and_stats():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(0x0, 0x100, "ram")
    assert sim._memif_access(sim.memif_code, 0x10, 4) == 0
    assert sim._memif_access(_BadMemIf(), 0x10, 4) == 0
    stats = sim._format_memif_stats("bad", _BadMemIf())
    assert "hits" in stats and "misses" in stats


def test_csr_read_write_and_mask():
    sim = RV32Sim()
    sim.supports_s_mode = False
    assert sim._mask_mstatus(0) & 0x1800 == 0x1800
    sim._csr_write(0xB00, 0x12)
    assert sim._csr_read(0xB00) == 0x12
    assert sim.mcycle_suppress is True
    sim._csr_write(0xB80, 0x34)
    assert sim._csr_read(0xB80) == 0x34
    sim._csr_write(0xB02, 0x56)
    assert sim._csr_read(0xB02) == 0x56
    sim._csr_write(0xB82, 0x78)
    assert sim._csr_read(0xB82) == 0x78
    sim._csr_write(0x3A0, 0x1f)
    assert sim._csr_read(0x3A0) == 0x1f
    sim._csr_write(0x3B0, 0x101)
    assert sim._csr_read(0x3B0) == (0x101 & ~0x1)
    sim._csr_write(0x300, 0x0)
    assert sim._csr_read(0x300) & 0x1800 == 0x1800
    sim._csr_write(0x301, 0x1)
    assert sim._csr_read(0x301) == sim.csrs[0x301]


def test_track_hw_access():
    sim = RV32Sim(detect_mode=True)
    sim.pc = 0x100
    sim._track_hw_access(0x40000000, "read/write", [0x12])
    entry = sim.hw_accesses[0x40000000]
    assert entry["type"] == "read/write"
    assert entry["count"] == 1
    assert "values_read" in entry
    assert "values_written" in entry


def test_stub_config_and_hw_access(tmp_path):
    sim = RV32Sim()
    cfg = {
        "timing": {"base_cycles": 2},
        "branch_predictor": {"enabled": False, "entries": 0},
        "memif": {"code": {"module": "memif:FlatMemIf", "config": {"penalty": 1}}},
        "memory_regions": [{"start": "0x0", "end": "0x100", "name": "ram"}],
        "hw_stubs": {"0x40000000": {"value": 1, "width": 4}},
        "func_stubs": {"0x200": 3},
        "tohost_addr": "0x80001000",
        "fromhost_addr": "0x80001008",
    }
    path = tmp_path / "stubs.json"
    path.write_text(json.dumps(cfg))
    sim.load_stub_config(str(path))
    assert sim.timing["base_cycles"] == 2
    assert sim.hw_stubs[0x40000000]["value"] == 1
    assert sim.func_stubs[0x200] == 3
    assert sim.tohost_addr == 0x80001000
    assert sim.hw_stubs[0x80001008]["width"] == 8

    sim.detect_mode = True
    sim._track_hw_access(0x40000000, "read", [0x12])
    out = tmp_path / "out.json"
    sim.save_hw_accesses(str(out))
    saved = json.loads(out.read_text())
    assert "hw_stubs" in saved


def test_stub_config_missing_file(capsys):
    sim = RV32Sim()
    sim.load_stub_config("missing.json")
    assert "No stub config found" in capsys.readouterr().out


def test_memif_load_failure_and_tohost_fallback():
    sim = RV32Sim()
    memif = sim._load_memif_from_cfg({"module": "nope:Missing", "config": {}}, "code")
    assert memif.name == "code"
    sim.tohost_addr = 0xdeadbeef
    sim.tohost_size = 4
    assert sim._read_tohost_value(0x55) == 0x55


def test_assisted_hw_access_hint():
    sim = RV32Sim(assisted_mode=True, detect_mode=True)
    sim.pc = 0x200
    sim._track_hw_access(0x40000010, "read", [0x1])
    assert "hint" in sim.hw_accesses[0x40000010]


def test_tohost_read_and_traps():
    sim = RV32Sim()
    sim.configure_tohost(0x0, size=4)
    sim._write_memory(0x0, (0x12345678).to_bytes(4, "little"))
    assert sim._read_tohost_value(0) == 0x12345678
    sim.last_instr = 0xdeadbeef
    with pytest.raises(TrapException):
        sim._illegal_instruction()


def test_step_returns_false_on_exception(monkeypatch):
    sim = RV32Sim()
    monkeypatch.setattr(sim, "execute", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sim.step() is False


def test_step_and_dump_regs(capsys, monkeypatch):
    sim = RV32Sim()
    monkeypatch.setattr(sim, "execute", lambda: None)
    assert sim.step() is True
    sim.dump_regs()
    out = capsys.readouterr().out
    assert "pc" in out


def test_bit_helpers_and_readonly_csrs():
    sim = RV32Sim()
    assert sim._ctz32(0) == 32
    assert sim._clz32(0) == 32
    sim._csr_write(0x7a0, 0x1234)
    assert sim._csr_read(0x7a0) == 0


def test_pmpaddr_read_masking():
    sim = RV32Sim()
    sim.pmpcfg0 = 0x18
    sim.pmpaddr0 = 0x5
    assert sim._pmpaddr0_read() == 0x4


def test_region_wrappers_and_handlers():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(0x0, 0x100, "ram")
    region = sim._find_region(0x0)
    assert region is not None
    sim._write_memory(0x0, b"\x01\x02")
    assert sim._read_memory(0x0, 2) == 0x0201
    assert sim._read_memory_bytes(0x0, 2) == b"\x01\x02"

    def reader(addr, size):
        return 0xFF

    def writer(addr, value, size):
        writer.last = (addr, value, size)

    sim.register_read_handler(0x1000, reader)
    sim.register_write_handler(0x1000, writer)
    assert sim._read_value(0x1000, 1) == 0xFF
    sim._write_value(0x1000, 1, 0xAA)
    assert writer.last[1] == 0xAA


def test_cli_paths(tmp_path, monkeypatch):
    svd_path = tmp_path / "tiny.svd"
    svd_path.write_text("<device><name>T</name><peripherals/></device>")
    assert_path = tmp_path / "assert.json"
    assert_path.write_text(json.dumps({"assertions": {}}))
    stub_path = tmp_path / "stubs.json"
    stub_path.write_text(json.dumps({"hw_stubs": {"0x40000000": 1}}))

    def no_sleep(_):
        raise KeyboardInterrupt()

    def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("gdb_server.GDBServer.start", noop)
    monkeypatch.setattr("gdb_server.GDBServer.stop", noop)
    monkeypatch.setattr("rtt_server.RTTServer.start_data", noop)
    monkeypatch.setattr("rtt_server.RTTServer.start_cmd", noop)
    monkeypatch.setattr("rtt_server.RTTServer.stop_data", noop)
    monkeypatch.setattr("rtt_server.RTTServer.stop_cmd", noop)
    monkeypatch.setattr("time.sleep", no_sleep)

    argv = [
        "rv32sim.py",
        "--rtt-openocd",
        "--detect",
        "--assisted",
        "--svd",
        str(svd_path),
        "--stub",
        str(stub_path),
        "--assert",
        str(assert_path),
        "--assert-verbose",
        "--assert-asm",
        "--assert-writes",
        "--mem-region=0x30000000:0x100:ram",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("rv32sim", run_name="__main__")


def test_cli_help_and_errors(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")

    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--nope"])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")

    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--assert"])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")

    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--mem-region=0x0:0"])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")


def test_branch_predictor_disabled_and_alias():
    sim = RV32Sim()
    sim._configure_branch_predictor({"enabled": False, "entries": 2})
    assert sim._bp_lookup(0x100) is None
    assert sim._bp_should_store(0x100, 0x200) is False
    sim._bp_insert(0x100, 0x200)

    sim._configure_branch_predictor(
        {"enabled": True, "entries": 2, "store_forward": True, "alias_word": True}
    )
    sim._bp_insert(0x100, 0x80)
    assert sim._bp_should_store(0x102, 0x200) is False


def test_memif_config_and_stats_paths():
    sim = RV32Sim()
    memif = sim._load_memif_from_cfg({"module": "", "config": {}}, "code")
    assert memif.name == "code"
    sim._configure_memifs(None)
    sim._configure_memifs("memif:FlatMemIf")
    assert sim._memif_access(None, 0x10, 4) == 0
    assert sim._memif_access(sim.memif_code, 0x40000000, 4) == 0

    class _EmptyStats:
        def stats(self):
            return {}

    assert sim._format_memif_stats("none", None) == ""
    assert sim._format_memif_stats("empty", _EmptyStats()) == ""


def test_mmio_wrapper_calls():
    sim = RV32Sim()
    sim.hw_stubs[0x40000000] = 1
    base, _, width = sim._find_stub_entry(0x40000000)
    assert base == 0x40000000
    assert width == 4

    handler = {"func": lambda *_args: 0x12, "width": 1}
    sim.read_handlers[0x50000000] = handler
    base, entry, _width = sim._find_handler_entry(0x50000000, sim.read_handlers)
    assert base == 0x50000000
    assert sim._call_read_handler(entry, 0x50000000, 1) == 0x12

    def writer(addr, value, size):
        writer.last = (addr, value, size)

    sim._call_write_handler({"func": writer, "width": 1}, 0, 0x34, 1)
    assert writer.last[1] == 0x34


def test_track_hw_access_write_and_assist():
    sim = RV32Sim(detect_mode=True, assisted_mode=True)
    write_addr = 0x40000000
    addr = 0x40000010
    sim.pc = 0x100
    sim._track_hw_access(write_addr, "write", 0x2)
    sim._track_hw_access(addr, "read/write", 0)
    sim._track_hw_access(addr, "read", 0x1)
    sim._track_hw_access(addr, "write", 0x2)
    entry = sim.hw_accesses[addr]
    assert entry["type"] == "read/write"
    assert entry["values_read"] == [0x1]
    assert entry["values_written"] == [0x2]
    assert "hint" in sim.hw_accesses[write_addr]


def test_load_stub_config_parsing_and_errors(tmp_path, capsys):
    sim = RV32Sim()
    cfg = {
        "timing": {"base_cycles": True, "load_penalty": "0x2"},
        "branch_predictor": {
            "enabled": "yes",
            "entries": "4",
            "store_forward": 1,
            "alias_word": "off",
        },
        "memif": "memif:FlatMemIf",
        "func_stubs": {"0x300": {"value": 7}},
    }
    path = tmp_path / "stubs.json"
    path.write_text(json.dumps(cfg))
    sim.load_stub_config(str(path))
    assert sim.timing["base_cycles"] == 1
    assert sim.timing["load_penalty"] == 2
    assert sim.bp_cfg["entries"] == 4
    assert sim.bp_cfg["alias_word"] is False
    assert sim.func_stubs[0x300] == 7

    bad = tmp_path / "bad.json"
    bad.write_text("{")
    sim.load_stub_config(str(bad))
    assert "Error loading stub config" in capsys.readouterr().out


def test_save_hw_accesses_hints_and_values(tmp_path):
    sim = RV32Sim()
    sim.hw_accesses = {
        0x40000000: {
            "type": "read/write",
            "count": 2,
            "pcs": {0x100},
            "values_read": [1],
            "values_written": [2],
            "hint": "note",
        }
    }
    out = tmp_path / "hw.json"
    sim.save_hw_accesses(str(out))
    saved = json.loads(out.read_text())
    entry = saved["hw_stubs"]["0x40000000"]
    assert "hint" in entry
    assert "values_read" in entry
    assert "values_written" in entry


def test_save_assert_config_wrapper(tmp_path):
    sim = RV32Sim()
    sim.assertions = {0x0: {"width": 4}}
    path = tmp_path / "assert.json"
    sim.save_assert_config(str(path))
    assert path.exists()


def test_load_elf_error_cases(tmp_path):
    sim = RV32Sim()
    bad = tmp_path / "bad.elf"
    bad.write_bytes(b"BAD")
    with pytest.raises(ValueError):
        sim.load_elf(str(bad))

    phdr = struct.pack("<IIIIIIII", 1, 0, 0, 0, 0, 0, 0, 0)
    bad_class = tmp_path / "badclass.elf"
    bad_class.write_bytes(_build_elf(phdr, e_ident_class=2))
    with pytest.raises(ValueError):
        sim.load_elf(str(bad_class))

    bad_phentsize = tmp_path / "badph.elf"
    bad_phentsize.write_bytes(_build_elf(phdr, phentsize=16))
    with pytest.raises(ValueError):
        sim.load_elf(str(bad_phentsize))

    bad_phnum = tmp_path / "badphnum.elf"
    bad_phnum.write_bytes(_build_elf(phdr, phnum=2))
    with pytest.raises(ValueError):
        sim.load_elf(str(bad_phnum))

    bad_machine = tmp_path / "badmach.elf"
    bad_machine.write_bytes(_build_elf(phdr, e_machine=3))
    with pytest.raises(ValueError):
        sim.load_elf(str(bad_machine))

    phdr_bad_sizes = struct.pack("<IIIIIIII", 1, 0, 0, 0, 8, 4, 0, 0)
    bad_sizes = tmp_path / "badsizes.elf"
    bad_sizes.write_bytes(_build_elf(phdr_bad_sizes))
    with pytest.raises(ValueError):
        sim.load_elf(str(bad_sizes))

    phdr_bad_offset = struct.pack("<IIIIIIII", 1, 0x200, 0, 0, 4, 4, 0, 0)
    bad_offset = tmp_path / "badoffset.elf"
    bad_offset.write_bytes(_build_elf(phdr_bad_offset))
    with pytest.raises(ValueError):
        sim.load_elf(str(bad_offset))


def test_load_elf_padding_and_paddr(tmp_path):
    sim = RV32Sim()
    p_offset = 0x100
    phdr = struct.pack("<IIIIIIII", 1, p_offset, 0x0, 0x100, 4, 8, 0, 0)
    data = _build_elf(phdr)
    if len(data) < p_offset:
        data += b"\x00" * (p_offset - len(data))
    data += b"\x01\x02\x03\x04"
    path = tmp_path / "ok.elf"
    path.write_bytes(data)
    sim.load_elf(str(path))
    assert sim._read_memory(0x0, 4) == 0x04030201
    assert sim._read_memory(0x4, 4) == 0
    assert sim._read_memory(0x100, 4) == 0x04030201


def test_wrapper_methods_and_tohost_fallback():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(0x20000000, 0x20000100, "ram")
    base = 0x20000000
    sim.store_word(base, 0x11223344)
    assert sim.load_half(base) == 0x3344
    sim.store_byte(base + 4, 0xAA)
    sim.store_half(base + 6, 0xBEEF)
    sim.store_word(base + 8, 0x55667788)
    value, _penalty = sim._load_memif(base, 4)
    assert value == 0x11223344
    sim._store_memif(base + 12, 4, 0x01020304)
    assert sim.load_word(base + 12) == 0x01020304
    sim.store_word(base, 0x00000013)
    sim.pc = base
    instr, _penalty = sim.fetch()
    assert instr != 0
    assert sim._read_tohost_value(0x55) == 0x55


def test_cli_option_parsing_variants(tmp_path, monkeypatch):
    svd_path = tmp_path / "tiny.svd"
    svd_path.write_text("<device><name>T</name><peripherals/></device>")
    stub_path = tmp_path / "stubs.json"
    stub_path.write_text(json.dumps({"hw_stubs": {}}))

    def no_sleep(_):
        raise KeyboardInterrupt()

    def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("gdb_server.GDBServer.start", noop)
    monkeypatch.setattr("gdb_server.GDBServer.stop", noop)
    monkeypatch.setattr("rtt_server.RTTServer.start_data", noop)
    monkeypatch.setattr("rtt_server.RTTServer.start_cmd", noop)
    monkeypatch.setattr("rtt_server.RTTServer.stop_data", noop)
    monkeypatch.setattr("rtt_server.RTTServer.stop_cmd", noop)
    monkeypatch.setattr("time.sleep", no_sleep)

    argv = [
        "rv32sim.py",
        "--port=4444",
        "--port",
        "4445",
        "--stub=" + str(stub_path),
        "--svd=" + str(svd_path),
        "--rtt",
        "--rtt-port=4000",
        "--rtt-port",
        "4001",
        "--rtt-cmd-port=5000",
        "--rtt-cmd-port",
        "5001",
        "--mem-region",
        "0x30000000:0x100:ram",
        "--assert-assist",
        "--assert-verbose=0",
        "--assert-out=assert.json",
        "--assert-out",
        "assert2.json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    globals_dict = runpy.run_module("rv32sim", run_name="__main__")
    globals_dict["sim"]._write_value(0x10000000, 1, ord("A"))


def test_cli_assert_assist_default_output(monkeypatch):
    def no_sleep(_):
        raise KeyboardInterrupt()

    def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("gdb_server.GDBServer.start", noop)
    monkeypatch.setattr("gdb_server.GDBServer.stop", noop)
    monkeypatch.setattr("time.sleep", no_sleep)
    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--assert-assist"])
    globals_dict = runpy.run_module("rv32sim", run_name="__main__")
    assert globals_dict["sim"].assert_save_path == "assertions.json"


def test_cli_assert_assist_explicit_output(monkeypatch):
    def no_sleep(_):
        raise KeyboardInterrupt()

    def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("gdb_server.GDBServer.start", noop)
    monkeypatch.setattr("gdb_server.GDBServer.stop", noop)
    monkeypatch.setattr("time.sleep", no_sleep)
    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--assert-assist=out.json"])
    globals_dict = runpy.run_module("rv32sim", run_name="__main__")
    assert globals_dict["sim"].assert_save_path == "out.json"


def test_cli_mem_region_missing_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "--mem-region=0x100"])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")


def test_cli_multiple_elfs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rv32sim.py", "a.elf", "b.elf"])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")


def test_cli_load_elf_error(tmp_path, monkeypatch):
    bad = tmp_path / "bad.elf"
    bad.write_bytes(b"BAD")
    monkeypatch.setattr(sys, "argv", ["rv32sim.py", str(bad)])
    with pytest.raises(SystemExit):
        runpy.run_module("rv32sim", run_name="__main__")


def test_cli_keyboard_interrupt_saves(monkeypatch):
    store = {}

    def stop_sleep(_):
        main_mod = sys.modules["__main__"]
        sim = main_mod.sim
        sim.assertions = {0x0: {"width": 4}}
        sim.assert_dirty = True
        sim.hw_accesses = {0x40000000: {"type": "read", "count": 1, "pcs": {0}}}

        def save_assert(path):
            store["assert_path"] = path

        def save_hw(path):
            store["hw_path"] = path

        sim.save_assert_config = save_assert
        sim.save_hw_accesses = save_hw
        raise KeyboardInterrupt()

    def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("gdb_server.GDBServer.start", noop)
    monkeypatch.setattr("gdb_server.GDBServer.stop", noop)
    monkeypatch.setattr("time.sleep", stop_sleep)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rv32sim.py",
            "--assert-assist",
            "--detect",
        ],
    )
    runpy.run_module("rv32sim", run_name="__main__", alter_sys=True)
    assert "assert_path" in store
    assert "hw_path" in store
