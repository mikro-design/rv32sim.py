from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rv32sim import HaltException, RV32Sim, TrapException

COREMARK_DIR = Path(__file__).resolve().parent / "coremark"
COREMARK_ELF = COREMARK_DIR / "build" / "coremark.elf"
DEFAULT_MAX_STEPS = 5_000_000


def _toolchain_prefix() -> str:
    return os.environ.get("CROSS_COMPILE", "riscv32-unknown-elf-")


def _skip_if_disabled() -> None:
    if os.environ.get("RV32SIM_SKIP_COREMARK"):
        pytest.skip("RV32SIM_SKIP_COREMARK is set")


def _ensure_build_tools() -> None:
    if shutil.which("make") is None:
        pytest.skip("make not available")
    prefix = _toolchain_prefix()
    if shutil.which(f"{prefix}gcc") is None:
        pytest.skip(f"missing toolchain: {prefix}gcc")


def _build_coremark() -> Path:
    try:
        subprocess.run(
            ["make", "-C", str(COREMARK_DIR)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"CoreMark build failed:\n{exc.stdout}\n{exc.stderr}")
    if not COREMARK_ELF.is_file():
        pytest.fail(f"CoreMark ELF not found at {COREMARK_ELF}")
    return COREMARK_ELF


def _max_steps() -> int:
    raw = os.environ.get("RV32SIM_COREMARK_MAX_STEPS")
    if raw is None:
        return DEFAULT_MAX_STEPS
    try:
        value = int(raw)
    except ValueError:
        pytest.fail(f"Invalid RV32SIM_COREMARK_MAX_STEPS value: {raw}")
    if value <= 0:
        pytest.fail(f"RV32SIM_COREMARK_MAX_STEPS must be > 0, got {value}")
    return value


def _run_coremark(elf_path: Path, max_steps: int) -> HaltException:
    sim = RV32Sim()
    sim.load_elf(str(elf_path))
    for _ in range(max_steps):
        try:
            sim.execute()
        except HaltException as exc:
            return exc
        except TrapException as exc:
            pytest.fail(
                f"CoreMark trapped: cause={exc.cause} tval=0x{exc.tval:08x}"
            )
    pytest.fail(f"CoreMark did not exit within {max_steps} steps")
    raise AssertionError("unreachable")


@pytest.fixture(scope="session")
def coremark_elf() -> Path:
    _skip_if_disabled()
    _ensure_build_tools()
    return _build_coremark()


def test_coremark_runs(coremark_elf: Path) -> None:
    exc = _run_coremark(coremark_elf, _max_steps())
    assert exc.reason == "exit"
    assert exc.code == 0


# Local Variables:
# eval: (blacken-mode)
# End:
