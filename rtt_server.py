import threading
import time
import shlex

from server_utils import run_server_loop, safe_close


class TelnetFilter:
    IAC = 0xFF
    DO = 0xFD
    DONT = 0xFE
    WILL = 0xFB
    WONT = 0xFC
    SB = 0xFA
    SE = 0xF0

    def __init__(self):
        self._state = "data"
        self._saw_cr = False

    def reset(self):
        self._state = "data"
        self._saw_cr = False

    def _handle_data_byte(self, byte, out):
        if self._saw_cr:
            if byte in (0x0A, 0x00):
                self._saw_cr = False
                return
            self._saw_cr = False
        if byte == 0x0D:
            out.append(0x0A)
            self._saw_cr = True
            return
        out.append(byte)

    def filter(self, data):
        out = bytearray()
        state = self._state
        for byte in data:
            if state == "data":
                if byte == self.IAC:
                    state = "iac"
                else:
                    self._handle_data_byte(byte, out)
            elif state == "iac":
                if byte == self.IAC:
                    self._handle_data_byte(self.IAC, out)
                    state = "data"
                elif byte == self.SB:
                    state = "sb"
                elif byte in (self.DO, self.DONT, self.WILL, self.WONT):
                    state = "iac_opt"
                else:
                    state = "data"
            elif state == "iac_opt":
                state = "data"
            elif state == "sb":
                if byte == self.IAC:
                    state = "sb_iac"
            elif state == "sb_iac":
                if byte == self.SE:
                    state = "data"
                elif byte == self.IAC:
                    state = "sb"
                else:
                    state = "sb"
        self._state = state
        return bytes(out)


class RTTServer:
    def __init__(self, sim):
        self.sim = sim
        self.port = None
        self.thread = None
        self.client = None
        self.running = False
        self.cmd_port = None
        self.cmd_thread = None
        self.cmd_client = None
        self.cmd_running = False
        self._data_telnet_filter = TelnetFilter()
        self._cmd_telnet_filter = TelnetFilter()
        self._server_socket = None
        self._cmd_server_socket = None
        self._cb_addr = None
        self._scan_start = None
        self._scan_size = None
        self._identifier = b"SEGGER RTT"
        self._last_scan = 0.0
        self._announced = False

    def _start_server(self, port, running_attr, port_attr, thread_attr, loop, label, stop_fn):
        running = getattr(self, running_attr)
        current_port = getattr(self, port_attr)
        if running and current_port == port:
            return
        if running and current_port != port:
            stop_fn()
        setattr(self, port_attr, port)
        setattr(self, running_attr, True)
        thread = threading.Thread(target=loop, daemon=True)
        setattr(self, thread_attr, thread)
        thread.start()
        print(f"[RTT] {label} listening on localhost:{port}")
        print("[RTT] Connect with:")
        print(f"    telnet 127.0.0.1 {port}")

    def _stop_server(self, running_attr, client_attr, server_attr):
        setattr(self, running_attr, False)
        client = getattr(self, client_attr)
        if client:
            safe_close(client)
            setattr(self, client_attr, None)
        server = getattr(self, server_attr)
        if server:
            safe_close(server)
            setattr(self, server_attr, None)

    def start_data(self, port=4444):
        self._start_server(
            port,
            "running",
            "port",
            "thread",
            self._server_loop,
            "Server",
            self.stop_data,
        )

    def stop_data(self):
        self._stop_server("running", "client", "_server_socket")

    def _server_loop(self):
        messages = {
            "bind_failed": "[RTT] Server bind failed: {error}",
            "client_connected": "[RTT] Client connected",
            "client_disconnected": "[RTT] Client disconnected",
            "server_error": "[RTT] Server error: {error}",
        }
        run_server_loop(
            "127.0.0.1",
            self.port,
            lambda: self.running,
            lambda value: setattr(self, "running", value),
            self._client_loop,
            messages,
            client_setter=lambda client: setattr(self, "client", client),
            server_setter=lambda server: setattr(self, "_server_socket", server),
            configure_client=lambda client: client.setblocking(False),
        )

    def _client_loop(self, client):
        self._data_telnet_filter.reset()
        while self.running:
            now = time.monotonic()
            if self._cb_addr is None and now - self._last_scan > 0.5:
                self._last_scan = now
                addr = self._find_control_block()
                if addr is not None:
                    self._cb_addr = addr
                    self._announced = False
            if self._cb_addr is not None:
                try:
                    data = self._read_up()
                except Exception:
                    self._cb_addr = None
                    data = b""
                if data:
                    try:
                        client.sendall(data)
                    except Exception:
                        break
            try:
                chunk = client.recv(4096)
                if chunk == b"":
                    break
                if chunk:
                    chunk = self._data_telnet_filter.filter(chunk)
                    if chunk:
                        self._write_down(chunk)
            except BlockingIOError:
                pass
            except Exception:
                break
            time.sleep(0.01)

    def start_cmd(self, port=4444):
        self._start_server(
            port,
            "cmd_running",
            "cmd_port",
            "cmd_thread",
            self._cmd_server_loop,
            "Command server",
            self.stop_cmd,
        )

    def stop_cmd(self):
        self._stop_server("cmd_running", "cmd_client", "_cmd_server_socket")

    def _cmd_server_loop(self):
        messages = {
            "bind_failed": "[RTT] Command server bind failed: {error}",
            "client_connected": "[RTT] Command client connected",
            "client_disconnected": "[RTT] Command client disconnected",
            "server_error": "[RTT] Command server error: {error}",
        }
        run_server_loop(
            "127.0.0.1",
            self.cmd_port,
            lambda: self.cmd_running,
            lambda value: setattr(self, "cmd_running", value),
            self._cmd_client_loop,
            messages,
            client_setter=lambda client: setattr(self, "cmd_client", client),
            server_setter=lambda server: setattr(self, "_cmd_server_socket", server),
            configure_client=lambda client: client.setblocking(False),
        )

    def _cmd_client_loop(self, client):
        buffer = b""
        self._cmd_telnet_filter.reset()
        self._cmd_send(client, "> ")
        while self.cmd_running:
            try:
                chunk = client.recv(4096)
                if chunk == b"":
                    break
                if chunk:
                    buffer += self._cmd_telnet_filter.filter(chunk)
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip().decode(errors="ignore")
                        if not line:
                            self._cmd_send(client, "> ")
                            continue
                        response = self._handle_command(line)
                        if response:
                            self._cmd_send(client, response)
                        self._cmd_send(client, "> ")
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except Exception:
                break

    def _cmd_send(self, client, text):
        try:
            client.sendall(text.encode())
        except Exception:
            pass

    def _handle_command(self, line):
        try:
            parts = shlex.split(line)
        except Exception:
            return "error: parse\n"
        if not parts:
            return ""
        if len(parts) >= 4 and parts[0] == "riscv" and parts[1] == "rtt" and parts[2] == "setup":
            try:
                addr = int(parts[3], 0)
                size = int(parts[4], 0) if len(parts) > 4 else 0
                identifier = parts[5] if len(parts) > 5 else "SEGGER RTT"
                if len(parts) > 6:
                    identifier = " ".join(parts[5:])
                self._scan_start = addr
                self._scan_size = size if size else None
                self._identifier = identifier.encode()
                self._cb_addr = None
                self._announced = False
                self._ensure_scan_region()
                return "OK\n"
            except Exception:
                return "error: rtt setup\n"
        if len(parts) >= 3 and parts[0] == "riscv" and parts[1] == "rtt" and parts[2] == "start":
            try:
                self._ensure_scan_region()
            except Exception:
                return "error: rtt start\n"
            addr = self._find_control_block()
            if addr is None:
                return "Control block not found\n"
            self._cb_addr = addr
            self._announced = False
            return f"Control block found at 0x{addr:08x}\n"
        if len(parts) >= 4 and parts[0] == "rtt" and parts[1] == "server" and parts[2] == "start":
            try:
                port = int(parts[3], 0)
            except Exception:
                return "error: rtt server start\n"
            self.start_data(port)
            return "OK\n"
        if len(parts) >= 3 and parts[0] == "rtt" and parts[1] == "server" and parts[2] == "stop":
            self.stop_data()
            return "OK\n"
        if parts[0] in ("exit", "quit"):
            self.stop_cmd()
            return "OK\n"
        return "error: unknown command\n"

    def _find_control_block(self):
        self.sim._ensure_memory_regions()
        signature = self._identifier or b"SEGGER RTT"
        if self._scan_start is not None and self._scan_size:
            try:
                data = self.sim._read_memory_bytes(self._scan_start, self._scan_size)
            except Exception:
                return None
            idx = data.find(signature)
            while idx != -1:
                addr = self._scan_start + idx
                if self._validate_cb(addr):
                    return addr
                idx = data.find(signature, idx + 1)
            return None
        for region in self.sim.memory_regions:
            data = region.data
            start = 0
            while True:
                idx = data.find(signature, start)
                if idx == -1:
                    break
                addr = region.start + idx
                if self._validate_cb(addr):
                    return addr
                start = idx + 1
        return None

    def _ensure_scan_region(self):
        if self._scan_start is None or not self._scan_size:
            return
        start = self._scan_start & 0xffffffff
        end = (self._scan_start + self._scan_size) & 0xffffffff
        if end <= start:
            raise ValueError("Invalid RTT scan range")
        region = self.sim._find_region(start)
        if region is None:
            self.sim._add_memory_region(start, end, "rtt")
            return
        if end > region.end:
            self.sim._expand_region(region, end)

    def _validate_cb(self, addr):
        try:
            ident = self.sim._read_memory_bytes(addr, 16)
        except Exception:
            return False
        signature = self._identifier or b"SEGGER RTT"
        if not ident.startswith(signature):
            return False
        num_up = self.sim._read_memory(addr + 16, 4)
        num_down = self.sim._read_memory(addr + 20, 4)
        if num_up <= 0 or num_up > 8 or num_down < 0 or num_down > 8:
            return False
        up = self._parse_buffer(addr + 24)
        down = self._parse_buffer(addr + 24 + 24)
        if not up or not down:
            return False
        if not self._announced:
            print(
                "[RTT] Found control block at 0x{addr:08x} "
                "(up buf 0x{up:08x} size {usize}, down buf 0x{down:08x} size {dsize})".format(
                    addr=addr,
                    up=up["buf_addr"],
                    usize=up["size"],
                    down=down["buf_addr"],
                    dsize=down["size"],
                )
            )
            self._announced = True
        return True

    def _parse_buffer(self, addr):
        buf_addr = self.sim._read_memory(addr + 4, 4)
        size = self.sim._read_memory(addr + 8, 4)
        wr = self.sim._read_memory(addr + 12, 4)
        rd = self.sim._read_memory(addr + 16, 4)
        if size == 0 or not self.sim._is_power_of_two(size):
            return None
        region = self.sim._find_region(buf_addr)
        if region is None or buf_addr + size > region.end:
            return None
        return {
            "buf_addr": buf_addr,
            "size": size,
            "wr": wr,
            "rd": rd,
            "wr_addr": addr + 12,
            "rd_addr": addr + 16,
        }

    def _read_up(self):
        if self._cb_addr is None:
            return b""
        up = self._parse_buffer(self._cb_addr + 24)
        if not up:
            self._cb_addr = None
            return b""
        wr = up["wr"] & 0xffffffff
        rd = up["rd"] & 0xffffffff
        if wr == rd:
            return b""
        size = up["size"]
        buf = up["buf_addr"]
        if wr > rd:
            data = self.sim._read_memory_bytes(buf + rd, wr - rd)
        else:
            first = self.sim._read_memory_bytes(buf + rd, size - rd)
            second = self.sim._read_memory_bytes(buf, wr)
            data = first + second
        self.sim._write_memory(up["rd_addr"], (wr & 0xffffffff).to_bytes(4, "little"))
        return data

    def _write_down(self, data):
        if self._cb_addr is None:
            return
        down = self._parse_buffer(self._cb_addr + 24 + 24)
        if not down:
            self._cb_addr = None
            return
        size = down["size"]
        buf = down["buf_addr"]
        wr = down["wr"] & 0xffffffff
        rd = down["rd"] & 0xffffffff
        for b in data:
            next_wr = (wr + 1) & (size - 1)
            if next_wr == rd:
                break
            self.sim._write_memory(buf + wr, bytes([b]))
            wr = next_wr
        self.sim._write_memory(down["wr_addr"], (wr & 0xffffffff).to_bytes(4, "little"))
