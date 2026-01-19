import collections
import sys

class SimpleUART:
    def __init__(self, sim, base=0x10000000):
        self.sim = sim
        self.base = base
        self.rx_fifo = collections.deque()
        self.tx_enabled = True
        self.rx_enabled = True
        
        # Offsets
        self.OFF_TXDATA = 0x00
        self.OFF_RXDATA = 0x04
        self.OFF_TXCTRL = 0x08
        self.OFF_RXCTRL = 0x0C
        
        # Register Handlers
        sim.register_read_handler(base + self.OFF_RXDATA, self._read_rxdata, width=4)
        sim.register_read_handler(base + self.OFF_TXCTRL, self._read_txctrl, width=4)
        sim.register_read_handler(base + self.OFF_RXCTRL, self._read_rxctrl, width=4)
        sim.register_write_handler(base + self.OFF_TXDATA, self._write_txdata, width=4)
        sim.register_write_handler(base + self.OFF_TXCTRL, self._write_txctrl, width=4)
        sim.register_write_handler(base + self.OFF_RXCTRL, self._write_rxctrl, width=4)

    def queue_input(self, text):
        if isinstance(text, str):
            text = text.encode('utf-8')
        for b in text:
            self.rx_fifo.append(b)

    def _read_txctrl(self):
        # Bit 0 is TX Ready. Always ready in simulation.
        val = 0
        if self.tx_enabled:
            val |= 1
        return val

    def _read_rxctrl(self):
        # Bit 0 is RX Ready (FIFO not empty)
        val = 0
        if self.rx_enabled and self.rx_fifo:
            val |= 1
        return val

    def _read_rxdata(self):
        if self.rx_fifo:
            return self.rx_fifo.popleft()
        return 0

    def _write_txdata(self, val):
        # Just print to stdout for now
        byte = val & 0xFF
        # Print char if printable, else hex
        if 32 <= byte < 127 or byte in (10, 13, 9):
            print(chr(byte), end='', flush=True)
        else:
            # print(f"<{byte:02x}>", end='', flush=True)
            pass

    def _write_txctrl(self, val):
        pass # Ignore writes for now

    def _write_rxctrl(self, val):
        pass # Ignore writes for now
