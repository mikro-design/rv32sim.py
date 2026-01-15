import binascii
import socket

from rv32sim import RV32Sim
from gdb_server import GDBServer


class _FakeSocket:
    def __init__(self, data=b""):
        self._data = bytearray(data)
        self.sent = bytearray()

    def recv(self, n, flags=0):
        if flags == socket.MSG_PEEK:
            return bytes(self._data[:n])
        if not self._data:
            return b""
        chunk = bytes(self._data[:n])
        del self._data[:n]
        return chunk

    def send(self, data):
        self.sent += data
        return len(data)

    def sendall(self, data):
        self.sent += data


def _make_rsp(payload):
    checksum = sum(payload.encode("latin-1")) % 256
    return f"${payload}#{checksum:02x}".encode("ascii")


def test_rsp_escape_roundtrip():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    payload = "a$b#c}d*e"
    escaped = gdb._escape_rsp(payload)
    assert gdb._unescape_rsp(escaped) == payload
    assert gdb._unescape_rsp("abc}") == "abc"


def test_memory_map_xml_contains_regions():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    sim._add_memory_region(0x0, 0x100, "flash")
    sim._add_memory_region(0x20000000, 0x20001000, "sram")
    gdb = GDBServer(sim)
    xml = gdb._memory_map_xml()
    assert 'start="0x00000000"' in xml
    assert 'length="0x100"' in xml
    assert 'start="0x20000000"' in xml


def test_ensure_write_region_adds_region():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    gdb = GDBServer(sim)
    gdb._ensure_write_region(0x3000, 0x20)
    region = sim._find_region(0x3000)
    assert region is not None
    assert region.start == 0x3000
    assert region.end >= 0x3020


def test_stop_reply_from_exception():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    assert gdb._stop_reply_from_exception(Exception("Program exited with code 7")) == "W07"
    assert gdb._stop_reply_from_exception(Exception("boom")) == "S05"


def test_recv_packet_ack_and_checksum():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    payload = "g"
    data = b"+" + _make_rsp(payload)
    sock = _FakeSocket(data)
    gdb.client = sock
    assert gdb._recv_packet() == payload
    assert b"+" in sock.sent


def test_recv_packet_bad_checksum_then_good():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    bad = b"$g#00"
    good = _make_rsp("m0,1")
    sock = _FakeSocket(bad + good)
    gdb.client = sock
    assert gdb._recv_packet() == "m0,1"
    assert b"-" in sock.sent


def test_check_interrupt_with_socketpair(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    sock = _FakeSocket(b"\x03")
    gdb.client = sock

    def fake_select(read, _w, _x, _t):
        return (read, [], [])

    monkeypatch.setattr("select.select", fake_select)
    assert gdb._check_interrupt() is True


def test_handle_session_packet_flow(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []

    def fake_send(data=""):
        replies.append(data)

    packets = [
        "?",
        "Hg0",
        "qSupported",
        "vMustReplyEmpty",
        "qXfer:features:read:target.xml:0,10",
        "qXfer:features:read:target.xml:ffff,10",
        "qXfer:features:read:target.xml:zz,10",
        "qXfer:memory-map:read::0,10",
        "qXfer:memory-map:read::zz,10",
        "g",
        "p0",
        "p20",
        "pGG",
        "P0=01000000",
        "P20=01000000",
        "Pbad",
        "G" + ("00" * (33 * 4)),
        "G" + ("00" * 10),
        "Gzz",
        "m0,4",
        "mzz,4",
        "M0,4:11223344",
        "M30000000,4:11223344",
        "M0,4:zz",
        "X0,4:ABCD",
        "X30000000,4:ABCD",
        "qRcmd," + binascii.hexlify(b"reset_counter").decode(),
        "qRcmd," + binascii.hexlify(b"show_stats").decode(),
        "qRcmd," + binascii.hexlify(b"run_steps 1").decode(),
        "qRcmd," + binascii.hexlify(b"run_until_pc 0x0 1").decode(),
        "qRcmd," + binascii.hexlify(b"load_elf missing.elf").decode(),
        "qRcmd," + binascii.hexlify(b"load_elf").decode(),
        "qRcmd," + binascii.hexlify(b"unknown_cmd").decode(),
        "s",
        "Z0,0,0",
        "z0,0,0",
        "D",
    ]

    feed = list(packets)

    def fake_recv():
        return feed.pop(0) if feed else None

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    monkeypatch.setattr(sim, "execute", lambda: None)

    gdb._handle_session(gdb.client)
    assert replies


def test_handle_session_qrcmd_error_paths(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []

    packets = [
        "qRcmd," + binascii.hexlify(b"run_steps 1").decode(),
        "qRcmd," + binascii.hexlify(b"run_until_pc 0x10 0").decode(),
        "k",
    ]
    feed = list(packets)

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    def raise_exec():
        raise Exception("Program exited with code 5")

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    monkeypatch.setattr(sim, "execute", raise_exec)

    gdb._handle_session(gdb.client)
    assert replies


def test_continue_hits_breakpoint(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    sim.pc = 0x100
    sim.breakpoints.add(0x100)
    replies = []
    feed = ["c", "k"]

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    monkeypatch.setattr(gdb, "_check_interrupt", lambda: False)
    monkeypatch.setattr(sim, "execute", lambda: None)

    gdb._handle_session(gdb.client)
    assert any(r.startswith("S") for r in replies)


def test_start_and_stop(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)

    class DummyThread:
        def __init__(self, target, daemon=True):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr("threading.Thread", DummyThread)
    gdb.start(1234)
    assert gdb.running is True
    gdb.stop()
    assert gdb.running is False


def test_handle_continue_and_step(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []
    feed = ["c", "s", "k"]

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    def raise_exc():
        raise Exception("Program exited with code 2")

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    monkeypatch.setattr(gdb, "_check_interrupt", lambda: True)
    monkeypatch.setattr(sim, "execute", raise_exc)

    gdb._handle_session(gdb.client)
    assert any(r.startswith("S") or r.startswith("W") or r == "OK" for r in replies)


def test_send_packet_no_client():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.client = None
    gdb._send_packet("OK")


def test_send_packet_with_client():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    sock = _FakeSocket()
    gdb.client = sock
    gdb._send_packet("OK")
    assert sock.sent.startswith(b"$")


def test_server_loop_bind_error(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True

    class BadServer:
        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            raise OSError("bind fail")

        def close(self):
            pass

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: BadServer())
    gdb._server_loop()
    assert gdb.running is False


def test_server_loop_accept_once(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    handled = {"count": 0}

    class FakeClient:
        def close(self):
            pass

    class FakeServer:
        def __init__(self):
            self.calls = 0

        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            pass

        def listen(self, _backlog):
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

    def fake_handle(_conn):
        handled["count"] += 1
        gdb.running = False

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: FakeServer())
    monkeypatch.setattr(gdb, "_handle_session", fake_handle)
    gdb._server_loop()
    assert handled["count"] == 1


def test_start_when_running_and_default_port(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.start()

    class DummyThread:
        def __init__(self, target, daemon=True):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

    gdb.running = False
    monkeypatch.setattr("threading.Thread", DummyThread)
    gdb.start()
    assert gdb.port == sim.gdb_port


def test_stop_closes_with_errors():
    sim = RV32Sim()
    gdb = GDBServer(sim)

    class BadClose:
        def close(self):
            raise RuntimeError("boom")

    gdb.client = BadClose()
    gdb._server_socket = BadClose()
    gdb.stop()


def test_ensure_write_region_zero_and_expand():
    sim = RV32Sim()
    sim.memory_regions = []
    sim._memory_initialized = True
    gdb = GDBServer(sim)
    gdb._ensure_write_region(0x0, 0)
    region = sim._add_memory_region(0x1000, 0x1010, "ram")
    gdb._ensure_write_region(0x1000, 0x20)
    assert region.end >= 0x1020


def test_recv_packet_edge_cases(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    assert gdb._recv_packet() is None

    gdb.client = _FakeSocket(b"")
    assert gdb._recv_packet() is None

    class BadSock:
        def recv(self, _n):
            raise RuntimeError("boom")

    gdb.client = BadSock()
    assert gdb._recv_packet() is None


def test_check_interrupt_error_paths(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)

    gdb.client = _FakeSocket(b"\x03")
    monkeypatch.setattr("select.select", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert gdb._check_interrupt() is False

    gdb.client = _FakeSocket(b"\x03")
    monkeypatch.setattr("select.select", lambda *_args, **_kwargs: ([], [], []))
    assert gdb._check_interrupt() is False

    class BadSock:
        def recv(self, _n, _flags=0):
            raise RuntimeError("boom")

    bad = BadSock()
    gdb.client = bad
    monkeypatch.setattr("select.select", lambda *_args, **_kwargs: ([bad], [], []))
    assert gdb._check_interrupt() is False

    empty = _FakeSocket(b"")
    gdb.client = empty
    monkeypatch.setattr("select.select", lambda *_args, **_kwargs: ([empty], [], []))
    assert gdb._check_interrupt() is False

    other = _FakeSocket(b"A")
    gdb.client = other
    monkeypatch.setattr("select.select", lambda *_args, **_kwargs: ([other], [], []))
    assert gdb._check_interrupt() is False


def test_stop_reply_parse_error():
    sim = RV32Sim()
    gdb = GDBServer(sim)
    assert gdb._stop_reply_from_exception(Exception("Program exited with code nope")) == "W00"


def test_handle_session_more_packets(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []

    packets = [
        "qXfer:features:read:target.xml:0,ffff",
        "qXfer:memory-map:read::ffff,10",
        "qXfer:memory-map:read::0,ffff",
        "p33",
        "P1=01000000",
        "P40=01000000",
        "M0,2:11223344",
        "X0,2:ABCD",
        "Zbad",
        "k",
    ]
    feed = list(packets)

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    gdb._handle_session(gdb.client)
    assert replies


def test_handle_session_memory_write_expand(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []
    packets = ["X30000000,4:ABCD", "k"]
    feed = list(packets)

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    calls = {"count": 0}
    real_write = sim._write_memory

    def flaky_write(addr, data):
        if calls["count"] == 0:
            calls["count"] += 1
            raise ValueError("boom")
        return real_write(addr, data)

    monkeypatch.setattr(sim, "_write_memory", flaky_write)
    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    gdb._handle_session(gdb.client)
    assert replies


def test_handle_session_qrcmd_run_until_pc_and_load(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []
    packets = [
        "qRcmd," + binascii.hexlify(b"run_until_pc 0x10 1").decode(),
        "qRcmd," + binascii.hexlify(b"load_elf ok.elf").decode(),
        "qRcmd,zz",
        "k",
    ]
    feed = list(packets)

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    monkeypatch.setattr(sim, "execute", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sim, "load_elf", lambda _path: None)
    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    gdb._handle_session(gdb.client)
    assert replies


def test_continue_stop_reply_none(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []
    feed = ["c", "k"]

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    def stop_exec():
        gdb.running = False
        gdb.client = None

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    monkeypatch.setattr(gdb, "_check_interrupt", lambda: False)
    monkeypatch.setattr(sim, "execute", stop_exec)
    gdb._handle_session(gdb.client)
    assert "S05" in replies


def test_continue_execute_error(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []
    feed = ["c", "k"]

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    monkeypatch.setattr(gdb, "_check_interrupt", lambda: False)
    monkeypatch.setattr(sim, "execute", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    gdb._handle_session(gdb.client)
    assert any(r.startswith("S") or r.startswith("W") for r in replies)


def test_breakpoint_parse_error(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True
    gdb.client = object()
    replies = []
    feed = ["Zbad", "k"]

    def fake_recv():
        return feed.pop(0) if feed else None

    def fake_send(data=""):
        replies.append(data)

    monkeypatch.setattr(gdb, "_recv_packet", fake_recv)
    monkeypatch.setattr(gdb, "_send_packet", fake_send)
    gdb._handle_session(gdb.client)
    assert "" in replies


def test_server_loop_timeout_and_close_error(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True

    class FakeClient:
        def close(self):
            raise RuntimeError("boom")

    class FakeServer:
        def __init__(self):
            self.calls = 0

        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            pass

        def listen(self, _backlog):
            pass

        def settimeout(self, _t):
            pass

        def accept(self):
            if self.calls == 0:
                self.calls += 1
                return FakeClient(), ("127.0.0.1", 0)
            gdb.running = False
            raise socket.timeout()

        def close(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: FakeServer())
    monkeypatch.setattr(gdb, "_handle_session", lambda _conn: None)
    gdb._server_loop()


def test_server_loop_error_print(monkeypatch):
    sim = RV32Sim()
    gdb = GDBServer(sim)
    gdb.running = True

    class FakeServer:
        def setsockopt(self, *_args, **_kwargs):
            pass

        def bind(self, _addr):
            pass

        def listen(self, _backlog):
            pass

        def settimeout(self, _t):
            pass

        def accept(self):
            gdb.running = False
            raise RuntimeError("boom")

        def close(self):
            pass

    monkeypatch.setattr("socket.socket", lambda *_args, **_kwargs: FakeServer())
    gdb._server_loop()
