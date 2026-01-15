import pytest

from rv32sim import HaltException, RV32Sim


def test_mmio_read_handler_masks_and_tracks():
    sim = RV32Sim()
    sim.detect_mode = True
    base = 0x40000000

    def reader(addr, size):
        return 0x11223344

    sim.register_read_handler(base, reader, width=4)
    assert sim._read_value(base + 1, 1) == 0x33
    assert sim._read_value(base + 2, 2) == 0x1122
    assert base in sim.hw_accesses
    assert sim.hw_accesses[base]["count"] >= 2


def test_mmio_write_handler_byte_fallback():
    sim = RV32Sim()
    base = 0x40000010
    writes = []

    def writer(addr, value, size):
        writes.append((addr, value, size))

    sim.register_write_handler(base, writer, width=1)
    sim._write_value(base, 4, 0xAABBCCDD)
    assert writes == [
        (base, 0xDD, 1),
        (base + 1, 0xCC, 1),
        (base + 2, 0xBB, 1),
        (base + 3, 0xAA, 1),
    ]


def test_mmio_stub_sequence_repeat():
    sim = RV32Sim()
    addr = 0x50000000
    sim.hw_stubs[addr] = {"sequence": [1, 2], "repeat": True}
    assert sim._read_value(addr, 4) == 1
    assert sim._read_value(addr, 4) == 2
    assert sim._read_value(addr, 4) == 1


def test_mmio_unmapped_reads_return_zero():
    sim = RV32Sim()
    sim.detect_mode = True
    addr = 0x60000000
    assert sim._read_value(addr, 4) == 0
    assert addr in sim.hw_accesses


def test_mmio_read_handler_byte_fallback():
    sim = RV32Sim()
    base = 0x40000100
    values = [0x11, 0x22, 0x33, 0x44]

    def reader(addr, size):
        return values[(addr - base) % len(values)]

    sim.register_read_handler(base, reader, width=1)
    assert sim._read_value(base, 4) == 0x44332211


def test_mmio_handler_typeerror_fallback():
    sim = RV32Sim()
    base = 0x40000200

    def reader():
        return 0xAABBCCDD

    def writer(value):
        writer.last = value

    sim.register_read_handler(base, reader, width=4)
    sim.register_write_handler(base, writer, width=4)
    assert sim._read_value(base, 4) == 0xAABBCCDD
    sim._write_value(base, 4, 0x11223344)
    assert writer.last == 0x11223344


def test_mmio_stub_update_and_hold_last():
    sim = RV32Sim()
    addr = 0x50000010
    sim.hw_stubs[addr] = {"sequence": [1, 2], "repeat": False, "hold_last": False, "value": 7}
    assert sim._read_value(addr, 4) == 1
    assert sim._read_value(addr, 4) == 2
    assert sim._read_value(addr, 4) == 7

    sim.hw_stubs[addr] = {"value": 0}
    sim._write_value(addr, 2, 0xABCD)
    assert sim.hw_stubs[addr]["value"] == 0xABCD


def test_mmio_stub_alignment_and_state_read(monkeypatch):
    sim = RV32Sim()
    base = 0x50000000
    sim.hw_stubs[base] = {"value": 0x11223344, "width": 4}
    assert sim._read_value(base + 2, 2) == 0x1122

    sim.assert_mode = True
    sim.hw_stubs.pop(base)
    sim.mmio_state[base] = {"width": 4, "value": 0xAABBCCDD}
    monkeypatch.setattr(sim, "_assert_read", lambda _a, _s: (None, None))
    assert sim._read_value(base, 4) == 0xAABBCCDD


def test_mmio_write_handler_partial_width():
    sim = RV32Sim()
    base = 0x40000300
    writes = []

    def writer(addr, value, size):
        writes.append((addr, value, size))

    sim.register_write_handler(base, writer, width=4)
    sim._write_value(base + 1, 2, 0xBEEF)
    assert writes == [
        (base + 1, 0xEF, 1),
        (base + 2, 0xBE, 1),
    ]


def test_mmio_assert_paths_and_to_host():
    sim = RV32Sim()
    base = 0x60000000
    sim.assert_mode = True
    sim.assert_writes = False
    sim.assertions = {base: {"width": 4, "read": {"value": 0x12345678}}}
    assert sim._read_value(base, 4) == 0x12345678

    sim.assertions[base]["write"] = {"value": 0x11223344}
    sim._write_value(base, 4, 0x11223344)

    sim.assert_writes = True
    sim._write_value(base, 4, 0x11223344)

    sim.configure_tohost(base, size=4)
    with pytest.raises(Exception):
        sim._write_value(base, 4, 0x1)


def test_mmio_stub_alignment_and_value_errors():
    sim = RV32Sim()
    base = 0x50000002
    sim.hw_stubs[base] = {"value": 0x1122, "width": 2}
    assert sim._read_value(base + 1, 1) == 0x11

    sim.hw_stubs[0x50000010] = {"value": "bad"}
    assert sim._read_value(0x50000010, 4) == 0

    sim.hw_stubs[0x50000014] = {"value": object()}
    assert sim._read_value(0x50000014, 4) == 0

    sim.hw_stubs[0x50000018] = {"sequence": [], "value": 7}
    assert sim._read_value(0x50000018, 4) == 7

    sim.hw_stubs[0x5000001c] = 0x55
    assert sim._read_value(0x5000001c, 1) == 0x55


def test_mmio_write_handlers_halt_on_tohost():
    sim = RV32Sim()
    base = 0x40001000
    sim.configure_tohost(base, size=4)
    writes = []

    def writer(addr, value, size):
        writes.append((addr, value, size))

    sim.register_write_handler(base, writer, width=4)
    with pytest.raises(HaltException):
        sim._write_value(base, 4, 0x1234)
    assert sim.tohost_value == 0x1234
    assert writes == [(base, 0x1234, 4)]

    sim = RV32Sim()
    base = 0x40002000
    sim.configure_tohost(base, size=2)
    writes = []

    def writer_byte(addr, value, size):
        writes.append((addr, value, size))

    sim.register_write_handler(base, writer_byte, width=1)
    with pytest.raises(HaltException):
        sim._write_value(base, 2, 0xAABB)
    assert sim.tohost_value == 0xAABB


def test_mmio_stub_write_halts_on_tohost():
    sim = RV32Sim()
    base = 0x50000040
    sim.configure_tohost(base, size=4)
    sim.hw_stubs[base] = 0
    with pytest.raises(HaltException):
        sim._write_value(base, 4, 0xAABBCCDD)
    assert sim.hw_stubs[base] == 0xAABBCCDD
    assert sim.tohost_value == 0xAABBCCDD
