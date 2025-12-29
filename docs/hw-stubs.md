# Hardware stub configuration

rv32sim loads JSON stub configuration files with --stub=FILE. The format is
intentionally small and allows memory layout, MMIO stubs, and function stubs to
be described in one place.

## Example

{
  "memory_regions": [
    {"name": "flash", "start": "0x00000000", "end": "0x00020000"},
    {"name": "sram", "start": "0x20000000", "end": "0x21000000"}
  ],
  "hw_stubs": {
    "0x10000000": {"value": 0, "width": 4},
    "0x10000004": {"sequence": [0, 1, 1, 2], "repeat": true}
  },
  "func_stubs": {
    "0x00001234": 0
  },
  "tohost_addr": "0x80001000",
  "tohost_size": 8,
  "fromhost_addr": "0x80001008"
}

See examples/hw_stubs.json for a minimal starter config.

## Fields

- memory_regions: Optional list of regions with start/end and an optional name.
  Values can be integers or hex strings. If omitted, defaults are used.
- hw_stubs: Map of address to a value or an object. Values can be integers or
  hex strings. Objects may include:
  - value: default value returned on reads.
  - width: number of bytes covered by the stub (default 4).
  - sequence: list of values to return in order on successive reads.
  - repeat: if true, loop through sequence entries forever.
  - hold_last: if true, keep returning the last sequence value after the list
    is exhausted (default true).
- func_stubs: Map of function entry address to return value. When the PC hits
  the entry, the simulator returns that value and skips the function body.
- tohost_addr/fromhost_addr: Optional addresses used by some test suites to
  signal completion.

When running with --detect or --assisted, rv32sim writes a stub template on
exit with additional fields like access_type and pc_locations. These fields are
safe to keep for documentation, but only value and width are required.
