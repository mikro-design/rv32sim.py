# Assertion JSON and SVD assistant

rv32sim can run in a strict MMIO assertion mode. When the program reads or
writes outside the configured memory regions, the simulator can either:

- Halt if the access is not covered by the assertion file, or
- Prompt for values and save them to an assertion JSON for later runs.

## CLI options

- `--assert=FILE`: load assertion JSON and enforce strict MMIO checks.
- `--assert-assist`: prompt on MMIO accesses and record answers.
- `--assert-verbose`: always show full field detail/enums in prompts.
- `--assert-asm`: show disassembly around MMIO accesses.
- `--assert-writes`: prompt/assert on MMIO writes (default: record only).
- `--assert-out=FILE`: set the output JSON name for `--assert-assist`.
- `--svd=FILE`: load a CMSIS-SVD file for register names, fields, and enums.

## Assertion JSON format

Top-level `assertions` is a map of register addresses to entry objects:

```json
{
  "assertions": {
    "0x40000000": {
      "width": 4,
      "register": "UART0.DATA",
      "read": { "value": "0x00000000" },
      "write": { "value": "0x00000041", "mask": "0x000000ff" }
    }
  }
}
```

Fields:

- `width`: register width in bytes (default 4).
- `register`/`peripheral`: optional labels (filled from SVD when available).
- `read`: value returned on reads. Can also be a sequence:
  `{"sequence": ["0x0", "0x1"], "repeat": true}`.
- `write`: expected write value and optional `mask`. Only masked bits are
  checked.
- `read`/`write` can include `"ignore": true` to allow accesses without checks.
- If `assertions` is missing, `hw_stubs` is accepted as read-only assertions.

## Prompt input grammar

When `--assert-assist` is enabled:

- Enter hex/decimal values: `0x1f`, `31`.
- Use field assignments with SVD: `FIELD=0x3`, `FIELD=ENUM`.
- Use `-` to ignore the access (no assertion).

Stop the simulator with Ctrl-C to write the assertion JSON.

Use the example JSON above as a starter and save it as `assertions.json`.

Note: SVD parsing is best-effort (direct registers/fields/enums). Derived
definitions may require manual overrides.
