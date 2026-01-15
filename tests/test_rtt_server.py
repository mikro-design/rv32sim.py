import socket
import threading

import pytest

from rv32sim import RV32Sim
from rtt_server import RTTServer, TelnetFilter


def _write_u32(sim, addr, value):
    sim._write_memory(addr, value.to_bytes(4, "little"))


def test_telnet_filter_strips_negotiation_and_cr():
    filt = TelnetFilter()
    data = bytes([0xFF, 0xFD, 0x01]) + b"hello\r\n"
    assert filt.filter(data) == b"hello\n"

    filt = TelnetFilter()
    assert filt.filter(bytes([0xFF, 0xFF])) == bytes([0xFF])


def test_rtt_control_block_read_write():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20000000
    sim._add_memory_region(base, base + 0x200, "rtt")

    cb_addr = base
    ident = b"SEGGER RTT"
    sim._write_memory(cb_addr, ident + b"\x00" * (16 - len(ident)))
    _write_u32(sim, cb_addr + 16, 1)
    _write_u32(sim, cb_addr + 20, 1)

    up_desc = cb_addr + 24
    down_desc = cb_addr + 24 + 24
    up_buf = base + 0x80
    down_buf = base + 0x90
    _write_u32(sim, up_desc + 4, up_buf)
    _write_u32(sim, up_desc + 8, 8)
    _write_u32(sim, up_desc + 12, 3)
    _write_u32(sim, up_desc + 16, 1)

    _write_u32(sim, down_desc + 4, down_buf)
    _write_u32(sim, down_desc + 8, 8)
    _write_u32(sim, down_desc + 12, 0)
    _write_u32(sim, down_desc + 16, 0)

    sim._write_memory(up_buf + 1, b"\x41\x42")

    rtt = RTTServer(sim)
    assert rtt._find_control_block() == cb_addr

    rtt._cb_addr = cb_addr
    data = rtt._read_up()
    assert data == b"\x41\x42"
    assert sim._read_memory(up_desc + 16, 4) == 3

    rtt._write_down(b"hi")
    assert sim._read_memory_bytes(down_buf, 2) == b"hi"
    assert sim._read_memory(down_desc + 12, 4) == 2


def test_rtt_setup_command_sets_scan_range():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    resp = rtt._handle_command("riscv rtt setup 0x1000 0x20 SEGGER RTT")
    assert resp == "OK\n"
    assert rtt._scan_start == 0x1000
    assert rtt._scan_size == 0x20


def test_telnet_filter_subnegotiation():
    filt = TelnetFilter()
    data = bytes([0xFF, 0xFA, 0x01, 0x02, 0xFF, 0xF0]) + b"ok"
    assert filt.filter(data) == b"ok"


def test_handle_command_errors(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    assert rtt._handle_command("riscv rtt setup bad") == "error: rtt setup\n"
    assert rtt._handle_command("riscv rtt start") == "Control block not found\n"
    assert rtt._handle_command("rtt server start bad") == "error: rtt server start\n"
    assert rtt._handle_command("rtt server stop") == "OK\n"
    assert rtt._handle_command("exit") == "OK\n"
    assert rtt._handle_command("nope") == "error: unknown command\n"

    def fail_scan():
        raise ValueError("bad")

    monkeypatch.setattr(rtt, "_ensure_scan_region", fail_scan)
    assert rtt._handle_command("riscv rtt start") == "error: rtt start\n"


def test_find_control_block_scan_range():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20001000
    sim._add_memory_region(base, base + 0x100, "rtt")
    ident = b"SEGGER RTT"
    sim._write_memory(base + 0x10, ident + b"\x00" * (16 - len(ident)))
    _write_u32(sim, base + 0x20, 1)
    _write_u32(sim, base + 0x24, 1)
    _write_u32(sim, base + 0x28 + 4, base + 0x80)
    _write_u32(sim, base + 0x28 + 8, 8)
    _write_u32(sim, base + 0x28 + 12, 0)
    _write_u32(sim, base + 0x28 + 16, 0)
    _write_u32(sim, base + 0x28 + 24 + 4, base + 0x90)
    _write_u32(sim, base + 0x28 + 24 + 8, 8)
    _write_u32(sim, base + 0x28 + 24 + 12, 0)
    _write_u32(sim, base + 0x28 + 24 + 16, 0)

    rtt = RTTServer(sim)
    rtt._scan_start = base
    rtt._scan_size = 0x100
    assert rtt._find_control_block() == base + 0x10


def test_parse_buffer_invalid_size():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20002000
    sim._add_memory_region(base, base + 0x100, "rtt")
    desc = base
    _write_u32(sim, desc + 4, base + 0x40)
    _write_u32(sim, desc + 8, 3)
    _write_u32(sim, desc + 12, 0)
    _write_u32(sim, desc + 16, 0)
    rtt = RTTServer(sim)
    assert rtt._parse_buffer(desc) is None


def test_start_stop_data_and_cmd(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    started = {"data": False, "cmd": False}

    class DummyThread:
        def __init__(self, target, daemon=True):
            self.target = target
            self.daemon = daemon

        def start(self):
            started["data"] = True

    class DummyCmdThread(DummyThread):
        def start(self):
            started["cmd"] = True

    monkeypatch.setattr("threading.Thread", DummyThread)
    rtt.start_data(1234)
    assert started["data"] is True
    rtt.stop_data()

    monkeypatch.setattr("threading.Thread", DummyCmdThread)
    rtt.start_cmd(1235)
    assert started["cmd"] is True
    rtt.stop_cmd()


def test_cmd_client_loop(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.cmd_running = True
    sent = []

    class FakeClient:
        def __init__(self, data):
            self._data = data

        def recv(self, _n):
            if self._data:
                data = self._data
                self._data = b""
                return data
            return b""

        def sendall(self, data):
            sent.append(data)

    monkeypatch.setattr(rtt, "_handle_command", lambda _line: "OK\n")
    client = FakeClient(b"rtt server stop\n")
    rtt._cmd_client_loop(client)
    assert sent


def test_server_loops(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.running = True
    rtt.cmd_running = True
    called = {"data": 0, "cmd": 0}

    class FakeClient:
        def setblocking(self, _flag):
            pass

        def close(self):
            pass

    class FakeServer:
        def __init__(self):
            self.calls = 0

        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            pass

        def listen(self, _n):
            pass

        def settimeout(self, _t):
            pass

        def accept(self):
            self.calls += 1
            if self.calls == 1:
                return FakeClient(), ("127.0.0.1", 0)
            raise socket.timeout()

        def close(self):
            pass

    def fake_client_loop(_client):
        called["data"] += 1
        rtt.running = False

    def fake_cmd_client_loop(_client):
        called["cmd"] += 1
        rtt.cmd_running = False

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: FakeServer())
    monkeypatch.setattr(rtt, "_client_loop", fake_client_loop)
    monkeypatch.setattr(rtt, "_cmd_client_loop", fake_cmd_client_loop)
    rtt._server_loop()
    rtt._cmd_server_loop()
    assert called["data"] == 1
    assert called["cmd"] == 1


def test_client_loop_reads(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.running = True
    rtt._cb_addr = None
    written = []

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def recv(self, _n):
            self.calls += 1
            if self.calls == 1:
                return b"hi"
            return b""

        def sendall(self, _data):
            pass

    monkeypatch.setattr(rtt, "_find_control_block", lambda: None)
    monkeypatch.setattr(rtt, "_write_down", lambda data: written.append(data))
    monkeypatch.setattr("time.sleep", lambda _t: None)
    rtt._client_loop(FakeClient())
    assert written == [b"hi"]


def test_start_data_reuse_and_stop(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    calls = {"stop": 0}

    def fake_stop():
        calls["stop"] += 1

    class DummyThread:
        def __init__(self, target, daemon=True):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

    monkeypatch.setattr("threading.Thread", DummyThread)
    monkeypatch.setattr(rtt, "stop_data", fake_stop)
    rtt.start_data(1234)
    rtt.start_data(1234)
    rtt.start_data(1235)
    assert calls["stop"] == 1


def test_stop_data_and_cmd_close():
    sim = RV32Sim()
    rtt = RTTServer(sim)

    class BadClient:
        def close(self):
            raise RuntimeError("boom")

    class BadServer:
        def close(self):
            raise RuntimeError("boom")

    rtt.client = BadClient()
    rtt._server_socket = BadServer()
    rtt.stop_data()

    rtt.cmd_client = BadClient()
    rtt._cmd_server_socket = BadServer()
    rtt.stop_cmd()


def test_server_bind_error(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.running = True

    class BadServer:
        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            raise OSError("bind fail")

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: BadServer())
    rtt._server_loop()
    assert rtt.running is False


def test_client_loop_with_cb_addr(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.running = True
    rtt._cb_addr = 0x1
    sent = {"count": 0}

    class FakeClient:
        def recv(self, _n):
            return b""

        def sendall(self, _data):
            sent["count"] += 1
            raise RuntimeError("send fail")

    monkeypatch.setattr(rtt, "_read_up", lambda: b"hi")
    monkeypatch.setattr("time.sleep", lambda _t: None)
    rtt._client_loop(FakeClient())
    assert sent["count"] == 1


def test_cmd_server_bind_error(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.cmd_running = True

    class BadServer:
        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            raise OSError("bind fail")

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: BadServer())
    rtt._cmd_server_loop()
    assert rtt.cmd_running is False


def test_cmd_send_failure():
    sim = RV32Sim()
    rtt = RTTServer(sim)

    class BadClient:
        def sendall(self, _data):
            raise RuntimeError("boom")

    rtt._cmd_send(BadClient(), "hello")


def test_ensure_scan_region_expand():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(0x1000, 0x1100, "rtt")
    rtt._scan_start = 0x1000
    rtt._scan_size = 0x200
    rtt._ensure_scan_region()
    region = sim._find_region(0x1000)
    assert region.end == 0x1200


def test_validate_cb_failures():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20003000
    sim._add_memory_region(base, base + 0x100, "rtt")
    sim._write_memory(base, b"BAD" + b"\x00" * 13)
    assert rtt._validate_cb(base) is False
    sim._write_memory(base, b"SEGGER RTT" + b"\x00" * (16 - len("SEGGER RTT")))
    _write_u32(sim, base + 16, 0)
    _write_u32(sim, base + 20, 0)
    assert rtt._validate_cb(base) is False


def test_read_up_wrap_and_empty():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20004000
    sim._add_memory_region(base, base + 0x100, "rtt")

    sim._write_memory(base, b"SEGGER RTT" + b"\x00" * (16 - len("SEGGER RTT")))
    _write_u32(sim, base + 16, 1)
    _write_u32(sim, base + 20, 1)
    up_desc = base + 24
    up_buf = base + 0x80
    _write_u32(sim, up_desc + 4, up_buf)
    _write_u32(sim, up_desc + 8, 8)
    _write_u32(sim, up_desc + 12, 2)
    _write_u32(sim, up_desc + 16, 6)
    sim._write_memory(up_buf + 6, b"\x41\x42")
    sim._write_memory(up_buf, b"\x43\x44")

    rtt._cb_addr = base
    data = rtt._read_up()
    assert data == b"\x41\x42\x43\x44"

    _write_u32(sim, up_desc + 12, 0)
    _write_u32(sim, up_desc + 16, 0)
    assert rtt._read_up() == b""


def test_write_down_full():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20005000
    sim._add_memory_region(base, base + 0x100, "rtt")
    sim._write_memory(base, b"SEGGER RTT" + b"\x00" * (16 - len("SEGGER RTT")))
    _write_u32(sim, base + 16, 1)
    _write_u32(sim, base + 20, 1)
    down_desc = base + 24 + 24
    down_buf = base + 0x80
    _write_u32(sim, down_desc + 4, down_buf)
    _write_u32(sim, down_desc + 8, 4)
    _write_u32(sim, down_desc + 12, 2)
    _write_u32(sim, down_desc + 16, 0)
    rtt._cb_addr = base
    rtt._write_down(b"hi")
    assert sim._read_memory_bytes(down_buf + 2, 1) == b"h"


def test_telnet_filter_additional_states():
    filt = TelnetFilter()
    assert filt.filter(b"\rX") == b"\nX"

    filt = TelnetFilter()
    data = bytes([0xFF, 0x00]) + b"A"
    assert filt.filter(data) == b"A"

    filt = TelnetFilter()
    data = bytes([0xFF, 0xFA, 0x01, 0xFF, 0xFF, 0x00, 0xFF, 0xF0])
    filt.filter(data)

    filt = TelnetFilter()
    data = bytes([0xFF, 0xFA, 0x01, 0xFF, 0x00])
    filt.filter(data)


def test_client_loop_scans_and_read_up_error(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.running = True
    rtt._cb_addr = None
    rtt._last_scan = 0.0

    monkeypatch.setattr("time.monotonic", lambda: 1.0)
    monkeypatch.setattr(rtt, "_find_control_block", lambda: 0x1000)
    monkeypatch.setattr(rtt, "_read_up", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    class FakeClient:
        def recv(self, _n):
            return b""

        def sendall(self, _data):
            pass

    rtt._client_loop(FakeClient())
    assert rtt._cb_addr is None


def test_client_loop_recv_errors(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.running = True
    rtt._cb_addr = None

    class BlockClient:
        def __init__(self):
            self.calls = 0

        def recv(self, _n):
            self.calls += 1
            if self.calls == 1:
                raise BlockingIOError()
            return b""

        def sendall(self, _data):
            pass

    monkeypatch.setattr("time.sleep", lambda _t: None)
    rtt._client_loop(BlockClient())

    class BadClient:
        def recv(self, _n):
            raise RuntimeError("boom")

        def sendall(self, _data):
            pass

    rtt._client_loop(BadClient())


def test_start_cmd_reuse_and_stop(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.cmd_running = True
    rtt.cmd_port = 1234
    rtt.start_cmd(1234)

    called = {"stop": 0}

    def fake_stop():
        called["stop"] += 1

    class DummyThread:
        def __init__(self, target, daemon=True):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

    monkeypatch.setattr("threading.Thread", DummyThread)
    monkeypatch.setattr(rtt, "stop_cmd", fake_stop)
    rtt.start_cmd(1235)
    assert called["stop"] == 1


def test_cmd_server_loop_errors(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.cmd_running = True

    class FakeClient:
        def setblocking(self, _flag):
            pass

        def close(self):
            raise RuntimeError("boom")

    class FakeServer:
        def __init__(self):
            self.calls = 0

        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            pass

        def listen(self, _n):
            pass

        def settimeout(self, _t):
            pass

        def accept(self):
            self.calls += 1
            if self.calls == 1:
                return FakeClient(), ("127.0.0.1", 0)
            rtt.cmd_running = False
            raise RuntimeError("boom")

        def close(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: FakeServer())
    monkeypatch.setattr(rtt, "_cmd_client_loop", lambda _client: None)
    rtt._cmd_server_loop()


def test_cmd_client_loop_edge_cases(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt.cmd_running = True
    sent = []

    class EmptyLineClient:
        def __init__(self):
            self.data = b"\n"

        def recv(self, _n):
            if self.data:
                data = self.data
                self.data = b""
                return data
            return b""

        def sendall(self, data):
            sent.append(data)

    rtt._cmd_client_loop(EmptyLineClient())
    assert sent

    class BlockClient:
        def __init__(self):
            self.calls = 0

        def recv(self, _n):
            self.calls += 1
            if self.calls == 1:
                raise BlockingIOError()
            return b""

        def sendall(self, _data):
            pass

    monkeypatch.setattr("time.sleep", lambda _t: None)
    rtt._cmd_client_loop(BlockClient())

    class BadClient:
        def recv(self, _n):
            raise RuntimeError("boom")

        def sendall(self, _data):
            pass

    rtt._cmd_client_loop(BadClient())


def test_handle_command_parse_and_start(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    assert rtt._handle_command('riscv rtt setup "bad') == "error: parse\n"
    assert rtt._handle_command("") == ""

    monkeypatch.setattr(rtt, "_ensure_scan_region", lambda: None)
    monkeypatch.setattr(rtt, "_find_control_block", lambda: 0x2000)
    resp = rtt._handle_command("riscv rtt start")
    assert "Control block found" in resp
    assert rtt._cb_addr == 0x2000

    called = {"start": 0}

    def fake_start(_port):
        called["start"] += 1

    monkeypatch.setattr(rtt, "start_data", fake_start)
    assert rtt._handle_command("rtt server start 1234") == "OK\n"
    assert called["start"] == 1


def test_find_control_block_error_and_scan(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt._scan_start = 0x1000
    rtt._scan_size = 0x10

    def boom(_addr, _size):
        raise RuntimeError("boom")

    monkeypatch.setattr(sim, "_read_memory_bytes", boom)
    assert rtt._find_control_block() is None

    data = b"SEGGER RTT" + b"\x00" * 6 + b"SEGGER RTT"
    monkeypatch.setattr(sim, "_read_memory_bytes", lambda _addr, _size: data)
    monkeypatch.setattr(rtt, "_validate_cb", lambda _addr: False)
    assert rtt._find_control_block() is None

    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20006000
    sim._add_memory_region(base, base + 0x40, "rtt")
    sim._write_memory(base, b"SEGGER RTT" + b"\x00" * (16 - len("SEGGER RTT")))
    rtt._scan_start = None
    rtt._scan_size = None
    assert rtt._find_control_block() is None


def test_ensure_scan_region_invalid_and_add():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    rtt._scan_start = 0xFFFFFFF0
    rtt._scan_size = 0x20
    with pytest.raises(ValueError):
        rtt._ensure_scan_region()

    sim.memory_regions = []
    sim._memory_initialized = True
    rtt._scan_start = 0x1000
    rtt._scan_size = 0x10
    rtt._ensure_scan_region()
    assert sim._find_region(0x1000) is not None


def test_validate_cb_exceptions(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    monkeypatch.setattr(sim, "_read_memory_bytes", lambda _addr, _size: (_ for _ in ()).throw(RuntimeError("boom")))
    assert rtt._validate_cb(0x0) is False

    sim = RV32Sim()
    rtt = RTTServer(sim)
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20007000
    sim._add_memory_region(base, base + 0x100, "rtt")
    sim._write_memory(base, b"SEGGER RTT" + b"\x00" * (16 - len("SEGGER RTT")))
    _write_u32(sim, base + 16, 1)
    _write_u32(sim, base + 20, 1)
    monkeypatch.setattr(rtt, "_parse_buffer", lambda _addr: None)
    assert rtt._validate_cb(base) is False


def test_parse_buffer_region_missing():
    sim = RV32Sim()
    rtt = RTTServer(sim)
    sim.memory_regions = []
    sim._memory_initialized = True
    base = 0x20008000
    sim._add_memory_region(base, base + 0x20, "rtt")
    desc = base
    _write_u32(sim, desc + 4, base + 0x100)
    _write_u32(sim, desc + 8, 8)
    _write_u32(sim, desc + 12, 0)
    _write_u32(sim, desc + 16, 0)
    assert rtt._parse_buffer(desc) is None


def test_read_up_and_write_down_edge_cases(monkeypatch):
    sim = RV32Sim()
    rtt = RTTServer(sim)
    assert rtt._read_up() == b""

    rtt._cb_addr = 0x1
    monkeypatch.setattr(rtt, "_parse_buffer", lambda _addr: None)
    assert rtt._read_up() == b""
    assert rtt._cb_addr is None

    rtt._cb_addr = None
    rtt._write_down(b"hi")

    rtt._cb_addr = 0x1
    monkeypatch.setattr(rtt, "_parse_buffer", lambda _addr: None)
    rtt._write_down(b"hi")
    assert rtt._cb_addr is None
