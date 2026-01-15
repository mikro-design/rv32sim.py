import binascii
import select
import socket
import struct
import threading

from server_utils import run_server_loop, safe_close

_TARGET_XML = """<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target>
  <architecture>riscv:rv32</architecture>
  <feature name="org.gnu.gdb.riscv.cpu">
    <reg name="zero" bitsize="32" regnum="0"/>
    <reg name="ra" bitsize="32" regnum="1"/>
    <reg name="sp" bitsize="32" regnum="2"/>
    <reg name="gp" bitsize="32" regnum="3"/>
    <reg name="tp" bitsize="32" regnum="4"/>
    <reg name="t0" bitsize="32" regnum="5"/>
    <reg name="t1" bitsize="32" regnum="6"/>
    <reg name="t2" bitsize="32" regnum="7"/>
    <reg name="s0" bitsize="32" regnum="8"/>
    <reg name="s1" bitsize="32" regnum="9"/>
    <reg name="a0" bitsize="32" regnum="10"/>
    <reg name="a1" bitsize="32" regnum="11"/>
    <reg name="a2" bitsize="32" regnum="12"/>
    <reg name="a3" bitsize="32" regnum="13"/>
    <reg name="a4" bitsize="32" regnum="14"/>
    <reg name="a5" bitsize="32" regnum="15"/>
    <reg name="a6" bitsize="32" regnum="16"/>
    <reg name="a7" bitsize="32" regnum="17"/>
    <reg name="s2" bitsize="32" regnum="18"/>
    <reg name="s3" bitsize="32" regnum="19"/>
    <reg name="s4" bitsize="32" regnum="20"/>
    <reg name="s5" bitsize="32" regnum="21"/>
    <reg name="s6" bitsize="32" regnum="22"/>
    <reg name="s7" bitsize="32" regnum="23"/>
    <reg name="s8" bitsize="32" regnum="24"/>
    <reg name="s9" bitsize="32" regnum="25"/>
    <reg name="s10" bitsize="32" regnum="26"/>
    <reg name="s11" bitsize="32" regnum="27"/>
    <reg name="t3" bitsize="32" regnum="28"/>
    <reg name="t4" bitsize="32" regnum="29"/>
    <reg name="t5" bitsize="32" regnum="30"/>
    <reg name="t6" bitsize="32" regnum="31"/>
    <reg name="pc" bitsize="32" type="code_ptr" regnum="32"/>
  </feature>
</target>"""


class GDBServer:
    def __init__(self, sim):
        self.sim = sim
        self.port = sim.gdb_port
        self.thread = None
        self.client = None
        self.running = False
        self._server_socket = None

    def start(self, port=None):
        if self.running:
            return
        if port is None:
            port = self.sim.gdb_port
        self.port = port
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
        print(f"[SIM] GDB server listening on localhost:{self.port}")
        print("[SIM] Connect with:")
        print("    riscv32-unknown-elf-gdb your_program.elf")
        print(f"    (gdb) target remote localhost:{self.port}")

    def stop(self):
        self.running = False
        if self.client:
            safe_close(self.client)
            self.client = None
        if self._server_socket:
            safe_close(self._server_socket)
            self._server_socket = None

    def _ensure_write_region(self, addr, length):
        if length <= 0:
            return
        region = self.sim._find_region(addr)
        if region is not None:
            end = addr + length
            if end > region.end:
                self.sim._expand_region(region, end)
            return
        start = addr
        end = addr + length
        self.sim._add_memory_region(start, end, "gdb")

    def _server_loop(self):
        messages = {
            "bind_failed": "[SIM] GDB server bind failed: {error}",
            "client_connected": "[SIM] GDB connected",
            "client_disconnected": "[SIM] GDB disconnected",
            "server_error": "[SIM] GDB server error: {error}",
        }
        run_server_loop(
            "127.0.0.1",
            self.port,
            lambda: self.running,
            lambda value: setattr(self, "running", value),
            self._handle_session,
            messages,
            client_setter=lambda client: setattr(self, "client", client),
            server_setter=lambda server: setattr(self, "_server_socket", server),
        )

    def _escape_rsp(self, data):
        escaped = []
        for c in data:
            if c in "$#}*":
                escaped.append("}")
                escaped.append(chr(ord(c) ^ 0x20))
            else:
                escaped.append(c)
        return "".join(escaped)

    def _unescape_rsp(self, data):
        out = []
        i = 0
        while i < len(data):
            if data[i] == "}":
                i += 1
                if i >= len(data):
                    break
                out.append(chr(ord(data[i]) ^ 0x20))
            else:
                out.append(data[i])
            i += 1
        return "".join(out)

    def _send_packet(self, data=""):
        if not self.client:
            return
        escaped = self._escape_rsp(data)
        checksum = sum(ord(c) for c in escaped) % 256
        packet = f"${escaped}#{checksum:02x}".encode("ascii")
        self.client.send(packet)

    def _memory_map_xml(self):
        self.sim._ensure_memory_regions()
        lines = [
            '<?xml version="1.0"?>',
            '<!DOCTYPE memory-map PUBLIC "+//IDN gnu.org//DTD GDB Memory Map V1.0//EN" '
            '"http://sourceware.org/gdb/gdb-memory-map.dtd">',
            "<memory-map>",
        ]
        for region in self.sim.memory_regions:
            start = region.start & 0xffffffff
            length = (region.end - region.start) & 0xffffffff
            lines.append(
                f'  <memory type="ram" start="0x{start:08x}" length="0x{length:x}"/>'
            )
        lines.append("</memory-map>")
        return "\n".join(lines)

    def _format_xfer_reply(self, payload, offset, length):
        if offset >= len(payload):
            return "l"
        chunk = payload[offset : offset + length]
        if offset + length >= len(payload):
            return "l" + chunk
        return "m" + chunk

    def _parse_xfer_request(self, pkt):
        parts = pkt.split(":")[-1].split(",")
        if len(parts) < 2:
            raise ValueError("Invalid qXfer request")
        return int(parts[0], 16), int(parts[1], 16)

    def _handle_qxfer_target(self, pkt):
        try:
            offset, length = self._parse_xfer_request(pkt)
            return self._format_xfer_reply(_TARGET_XML, offset, length)
        except Exception as e:
            print(f"[SIM] qXfer parse error: {e}")
            return "E01"

    def _handle_qxfer_memory_map(self, pkt):
        memory_map = self._memory_map_xml()
        try:
            offset, length = self._parse_xfer_request(pkt)
            return self._format_xfer_reply(memory_map, offset, length)
        except Exception as e:
            print(f"[SIM] qXfer memory-map parse error: {e}")
            return "E01"

    def _handle_read_all_registers(self):
        reg_bytes = (
            b"".join(struct.pack("<I", r) for r in self.sim.regs)
            + struct.pack("<I", self.sim.pc)
        )
        return binascii.hexlify(reg_bytes).decode()

    def _handle_read_register(self, pkt):
        try:
            idx = int(pkt[1:], 16)
            if idx < 32:
                val = self.sim.regs[idx]
                return binascii.hexlify(struct.pack("<I", val & 0xffffffff)).decode()
            if idx == 32:
                val = self.sim.pc
                return binascii.hexlify(struct.pack("<I", val & 0xffffffff)).decode()
            return "E01"
        except Exception:
            return ""

    def _handle_write_register(self, pkt):
        try:
            reg_str, val_hex = pkt[1:].split("=", 1)
            reg = int(reg_str, 16)
            val_bytes = bytes.fromhex(val_hex)
            value = int.from_bytes(val_bytes, "little")
            if reg == 0:
                return "OK"
            if reg < 32:
                self.sim.regs[reg] = value & 0xffffffff
                return "OK"
            if reg == 32:
                self.sim.pc = value & 0xffffffff
                return "OK"
            return "E01"
        except Exception as e:
            print(f"[SIM] Register write error: {e}")
            return "E01"

    def _handle_write_all_registers(self, pkt):
        try:
            data = bytes.fromhex(pkt[1:])
            reg_count = 33
            needed = reg_count * 4
            if len(data) < needed:
                return "E01"
            for i in range(32):
                self.sim.regs[i] = (
                    int.from_bytes(data[i * 4 : (i + 1) * 4], "little")
                    & 0xffffffff
                )
            self.sim.regs[0] = 0
            self.sim.pc = int.from_bytes(data[32 * 4 : 33 * 4], "little") & 0xffffffff
            return "OK"
        except Exception as e:
            print(f"[SIM] Register block write error: {e}")
            return "E01"

    def _handle_read_memory(self, pkt):
        try:
            parts = pkt[1:].split(",")
            addr = int(parts[0], 16)
            length = int(parts[1], 16)
            data = bytes(self.sim.load_byte(addr + i) for i in range(length))
            return binascii.hexlify(data).decode()
        except Exception:
            return "E01"

    def _handle_write_memory_hex(self, pkt):
        try:
            header, data_hex = pkt[1:].split(":", 1)
            addr_str, length_str = header.split(",", 1)
            addr = int(addr_str, 16)
            length = int(length_str, 16)
            data = bytes.fromhex(data_hex)
            if len(data) != length:
                data = data[:length]
            try:
                self.sim._write_memory(addr, data)
            except ValueError:
                self._ensure_write_region(addr, len(data))
                self.sim._write_memory(addr, data)
            return "OK"
        except Exception as e:
            print(f"[SIM] Memory write error: {e}")
            return "E01"

    def _handle_write_memory_bin(self, pkt):
        try:
            header, data_raw = pkt[1:].split(":", 1)
            addr_str, length_str = header.split(",", 1)
            addr = int(addr_str, 16)
            length = int(length_str, 16)
            data = data_raw.encode("latin-1")
            if len(data) != length:
                data = data[:length]
            try:
                self.sim._write_memory(addr, data)
            except ValueError:
                self._ensure_write_region(addr, len(data))
                self.sim._write_memory(addr, data)
            return "OK"
        except Exception as e:
            print(f"[SIM] Binary memory write error: {e}")
            return "E01"

    def _handle_remote_command(self, cmd):
        if cmd == "reset_counter":
            self.sim.instr_count = 0
            self.sim.mcycle = 0
            self.sim.minstret = 0
            self.sim._reset_perf_state()
            response = "Instruction/cycle counters reset\n"
            return binascii.hexlify(response.encode()).decode()
        if cmd == "show_stats":
            response = (
                f"Instructions executed: {self.sim.instr_count}\n"
                f"Cycles executed: {self.sim.mcycle}\n"
            )
            response += self.sim._format_memif_stats("code", self.sim.memif_code)
            response += self.sim._format_memif_stats("data", self.sim.memif_data)
            return binascii.hexlify(response.encode()).decode()
        if cmd.startswith("run_steps"):
            parts = cmd.split()
            steps = int(parts[1], 0) if len(parts) > 1 else 0
            executed = 0
            error = None
            while executed < steps:
                try:
                    self.sim.execute()
                except Exception as e:
                    error = e
                    break
                executed += 1
            if error:
                response = (
                    f"Stopped after {executed} steps: {error}\n"
                    f"PC=0x{self.sim.pc:08x}\n"
                )
            else:
                response = f"Ran {executed} steps\nPC=0x{self.sim.pc:08x}\n"
            return binascii.hexlify(response.encode()).decode()
        if cmd.startswith("run_until_pc"):
            parts = cmd.split()
            target = int(parts[1], 0) if len(parts) > 1 else self.sim.pc
            max_steps = int(parts[2], 0) if len(parts) > 2 else 1000000
            executed = 0
            error = None
            while executed < max_steps and self.sim.pc != target:
                try:
                    self.sim.execute()
                except Exception as e:
                    error = e
                    break
                executed += 1
            if error:
                response = (
                    f"Stopped after {executed} steps: {error}\n"
                    f"PC=0x{self.sim.pc:08x}\n"
                )
            elif self.sim.pc == target:
                response = (
                    f"Hit PC=0x{target:08x} after {executed} steps\n"
                    f"PC=0x{self.sim.pc:08x}\n"
                )
            else:
                response = (
                    f"Max steps reached ({executed}) without hitting PC=0x{target:08x}\n"
                    f"PC=0x{self.sim.pc:08x}\n"
                )
            return binascii.hexlify(response.encode()).decode()
        if cmd.startswith("load_elf"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                response = "Usage: load_elf <path>\n"
            else:
                elf_path = parts[1].strip()
                try:
                    self.sim.load_elf(elf_path)
                    response = f"Loaded ELF {elf_path}\nPC=0x{self.sim.pc:08x}\n"
                except Exception as e:
                    response = f"Failed to load ELF: {e}\n"
            return binascii.hexlify(response.encode()).decode()
        return ""

    def _handle_continue(self):
        stop_reply = None
        while self.client and self.running:
            if self._check_interrupt():
                stop_reply = "S02"
                break
            if self.sim.pc in self.sim.breakpoints:
                if self.sim._skip_breakpoint_once:
                    self.sim._skip_breakpoint_once = False
                else:
                    print(f"[SIM] Hit breakpoint at PC=0x{self.sim.pc:08x}")
                    self.sim._skip_breakpoint_once = True
                    stop_reply = "S05"
                    break
            try:
                self.sim.execute()
            except Exception as e:
                print(f"[SIM] Exception at PC={self.sim.pc:08x}: {e}")
                stop_reply = self._stop_reply_from_exception(e)
                break
        if stop_reply is None:
            stop_reply = "S05"
        self._send_packet(stop_reply)

    def _handle_step(self):
        try:
            self.sim.execute()
            return "S05"
        except Exception as e:
            return self._stop_reply_from_exception(e)

    def _handle_breakpoint(self, pkt):
        try:
            comma_idx = pkt.index(",")
            parts = pkt[comma_idx + 1 :].split(",")
            addr = int(parts[0], 16)
            if pkt[0] == "Z":
                self.sim.add_breakpoint(addr)
            else:
                self.sim.remove_breakpoint(addr)
            return "OK"
        except Exception as e:
            print(f"[SIM] Breakpoint parse error: {e}")
            return ""

    def _recv_packet(self):
        if not self.client:
            return None
        state = 0
        data = ""
        checksum = 0
        while True:
            try:
                b = self.client.recv(1)
                if not b:
                    return None
                c = b.decode("latin-1")
                if state == 0:
                    if c in "+-":
                        continue
                    if c == "$":
                        state = 1
                        data = ""
                        checksum = 0
                elif state == 1:
                    if c == "#":
                        state = 2
                    else:
                        data += c
                        checksum = (checksum + ord(c)) % 256
                elif state == 2:
                    chk_hi = int(c, 16) << 4
                    state = 3
                elif state == 3:
                    if (chk_hi + int(c, 16)) % 256 == checksum:
                        self.client.send(b"+")
                        return self._unescape_rsp(data)
                    self.client.send(b"-")
                    state = 0
            except Exception:
                return None

    def _check_interrupt(self):
        if not self.client:
            return False
        try:
            ready, _, _ = select.select([self.client], [], [], 0)
        except Exception:
            return False
        if not ready:
            return False
        try:
            data = self.client.recv(1, socket.MSG_PEEK)
        except Exception:
            return False
        if not data:
            return False
        if data == b"\x03":
            try:
                self.client.recv(1)
            except Exception:
                pass
            return True
        return False

    def _stop_reply_from_exception(self, exc):
        msg = str(exc)
        if "Program exited with code" in msg:
            try:
                code = int(msg.split("code", 1)[1].strip().split()[0], 0) & 0xff
            except Exception:
                code = 0
            return f"W{code:02x}"
        return "S05"

    def _handle_session(self, conn):
        # Don't send unsolicited packet - wait for GDB to query with '?'
        while self.running:
            pkt = self._recv_packet()
            if pkt is None:
                break

            reply = ""

            if pkt == "?":
                reply = "S05"

            elif pkt in ("D", "k"):
                reply = "OK"
                self._send_packet(reply)
                break

            elif pkt.startswith("Hg") or pkt.startswith("Hc"):
                reply = "OK"

            elif "qSupported" in pkt:
                reply = "PacketSize=1000;qXfer:features:read+;qXfer:memory-map:read+"

            elif pkt == "vMustReplyEmpty" or pkt.startswith("v"):
                reply = ""

            elif pkt.startswith("qXfer:features:read:target.xml:"):
                reply = self._handle_qxfer_target(pkt)

            elif pkt.startswith("qXfer:memory-map:read::"):
                reply = self._handle_qxfer_memory_map(pkt)

            elif pkt == "g":
                reply = self._handle_read_all_registers()

            elif pkt.startswith("p"):
                reply = self._handle_read_register(pkt)

            elif pkt.startswith("P"):
                reply = self._handle_write_register(pkt)

            elif pkt.startswith("G"):
                reply = self._handle_write_all_registers(pkt)

            elif pkt.startswith("m"):
                reply = self._handle_read_memory(pkt)

            elif pkt.startswith("M"):
                reply = self._handle_write_memory_hex(pkt)

            elif pkt.startswith("X"):
                reply = self._handle_write_memory_bin(pkt)

            elif pkt.startswith("qRcmd,"):
                try:
                    cmd = bytes.fromhex(pkt[6:]).decode("ascii").strip()
                    reply = self._handle_remote_command(cmd)
                except Exception:
                    reply = ""

            elif pkt == "c":
                self._handle_continue()
                continue

            elif pkt.startswith("s"):
                reply = self._handle_step()

            elif pkt.startswith("Z") or pkt.startswith("z"):
                reply = self._handle_breakpoint(pkt)

            self._send_packet(reply)
