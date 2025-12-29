# Test ELF binaries

This repo includes prebuilt ELF binaries from three upstream RISC-V test
suites. Only the binaries are included; upstream source trees are not part of
this repository.

## Locations

- tests/elfs/riscv-tests
  - ISA test binaries from the riscv-tests suite.
- tests/elfs/riscv-arch-test
  - Binaries produced by the riscv-arch-test suite.
- tests/elfs/riscv-torture
  - A riscv-torture stress test binary.

## Licenses

Upstream licenses are included in third_party/. If you redistribute the test
binaries, follow the terms in those files.
