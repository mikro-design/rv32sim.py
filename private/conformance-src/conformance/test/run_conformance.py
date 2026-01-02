#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rv32sim import RV32Sim, HaltException, TrapException

DEFAULT_SIG_BEGIN = ["begin_signature", "rvtest_sig_begin", "signature_begin"]
DEFAULT_SIG_END = ["end_signature", "rvtest_sig_end", "signature_end"]


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def run_cmd(cmd, cwd=None, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def java_major_version(java_cmd):
    try:
        out = subprocess.check_output([java_cmd, "-version"], stderr=subprocess.STDOUT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    match = re.search(r'version "([^"]+)"', out)
    if not match:
        return None
    ver = match.group(1)
    if ver.startswith("1."):
        try:
            return int(ver.split(".")[1])
        except (IndexError, ValueError):
            return None
    try:
        return int(ver.split(".")[0])
    except ValueError:
        return None


def toolchain_paths(config):
    prefix = config.get("toolchain_prefix")
    if prefix:
        gcc_path = config.get("gcc_path", prefix + "gcc")
        nm_path = config.get("nm_path", prefix + "nm")
    else:
        gcc_path = config.get("gcc_path", "riscv32-unknown-elf-gcc")
        nm_path = config.get("nm_path", "riscv32-unknown-elf-nm")
    return gcc_path, nm_path, prefix


def get_symbols(nm_path, elf_path):
    sym = {}
    try:
        out = subprocess.check_output([nm_path, "-n", str(elf_path)], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sym
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        addr_str, _, name = parts[0], parts[1], parts[2]
        try:
            addr = int(addr_str, 16)
        except ValueError:
            continue
        sym[name] = addr
    return sym


def pick_symbol(sym, names):
    for name in names:
        if name in sym:
            return sym[name]
    return None


def find_section_addr(elf_path, section_name):
    try:
        data = Path(elf_path).read_bytes()
    except OSError:
        return None
    if len(data) < 52 or data[:4] != b"\x7fELF":
        return None
    if data[4] != 1 or data[5] != 1:
        return None  # only 32-bit little-endian supported

    e_shoff = struct.unpack_from("<I", data, 32)[0]
    e_shentsize = struct.unpack_from("<H", data, 46)[0]
    e_shnum = struct.unpack_from("<H", data, 48)[0]
    e_shstrndx = struct.unpack_from("<H", data, 50)[0]
    if e_shoff == 0 or e_shnum == 0 or e_shentsize < 40:
        return None
    if e_shstrndx >= e_shnum:
        return None
    shstr_off = e_shoff + e_shstrndx * e_shentsize
    if shstr_off + e_shentsize > len(data):
        return None
    sh = struct.unpack_from("<IIIIIIIIII", data, shstr_off)
    sh_offset, sh_size = sh[4], sh[5]
    if sh_offset + sh_size > len(data):
        return None
    shstr = data[sh_offset:sh_offset + sh_size]
    target = section_name.encode("ascii", errors="ignore")

    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if off + e_shentsize > len(data):
            break
        sh = struct.unpack_from("<IIIIIIIIII", data, off)
        sh_name, sh_addr = sh[0], sh[3]
        if sh_name >= len(shstr):
            continue
        end = shstr.find(b"\x00", sh_name)
        if end == -1:
            continue
        if shstr[sh_name:end] == target:
            return sh_addr
    return None


def dump_signature(sim, begin, end, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for addr in range(begin, end, 4):
            word = sim.load_word(addr)
            f.write(f"{word:08x}\n")


def parse_makefrag_tests(makefrag_path, group):
    tests = []
    capturing = False
    with open(makefrag_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not capturing:
                if stripped.startswith(f"{group}_sc_tests"):
                    capturing = True
                    _, rest = stripped.split("=", 1)
                    line_part = rest.strip()
                else:
                    continue
            else:
                line_part = stripped
            cont = line_part.endswith("\\")
            line_part = line_part.rstrip("\\").strip()
            if line_part:
                tests.extend(line_part.split())
            if not cont:
                break
    return tests


def apply_sim_config(sim, config, config_path):
    if config_path:
        sim.load_stub_config(config_path)
    regions = config.get("memory_regions")
    if regions and not sim.memory_regions:
        sim.memory_regions = []
        sim._memory_initialized = True
        for region in regions:
            start = int(region["start"], 16) if isinstance(region["start"], str) else region["start"]
            end = int(region["end"], 16) if isinstance(region["end"], str) else region["end"]
            name = region.get("name", "unknown")
            sim._add_memory_region(start, end, name)


def run_elf(elf_path, config, config_path, nm_path, suite_name):
    sim = RV32Sim()
    apply_sim_config(sim, config, config_path)
    sim.load_elf(str(elf_path))
    sim.regs[2] = (sim.get_stack_top() - 16) & 0xffffffff
    sym = get_symbols(nm_path, elf_path)
    if sim.tohost_addr is None:
        tohost = config.get("tohost_addr")
        if tohost is None:
            tohost = pick_symbol(sym, ["tohost"])
        if tohost is None:
            tohost = find_section_addr(elf_path, ".tohost")
        if tohost is not None:
            sim.configure_tohost(tohost, size=config.get("tohost_size", 8))

    signature_begin = None
    signature_end = None
    sig_cfg = config.get("signature", {})
    begin_names = sig_cfg.get("begin_symbols", DEFAULT_SIG_BEGIN)
    end_names = sig_cfg.get("end_symbols", DEFAULT_SIG_END)
    signature_begin = pick_symbol(sym, begin_names)
    signature_end = pick_symbol(sym, end_names)

    max_steps = int(config.get("max_instructions", 5_000_000))
    progress_interval = int(config.get("progress_interval", 0) or 0)
    progress_seconds = float(config.get("progress_seconds", 0) or 0)
    next_progress = progress_interval if progress_interval > 0 else None
    status = "timeout"
    reason = "max_instructions"
    code = None
    error = None

    sim.instr_count = 0
    if progress_interval > 0:
        print(f"[SIM] Progress interval: {progress_interval} instructions", flush=True)
    stop_event = None
    progress_thread = None
    if progress_seconds > 0:
        print(f"[SIM] Progress interval: {progress_seconds:.1f}s", flush=True)
        stop_event = threading.Event()

        def progress_loop():
            last_count = sim.instr_count
            last_time = time.time()
            while not stop_event.wait(progress_seconds):
                now = time.time()
                count = sim.instr_count
                delta = count - last_count
                rate = delta / (now - last_time) if now > last_time else 0.0
                stalled = " (stalled)" if delta == 0 else ""
                print(
                    f"[SIM] Progress: {count}/{max_steps} PC=0x{sim.pc:08x} {rate:.0f} inst/s{stalled}",
                    flush=True,
                )
                last_count = count
                last_time = now

        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.start()

    try:
        while sim.instr_count < max_steps:
            try:
                sim.execute()
                if next_progress is not None and sim.instr_count >= next_progress:
                    print(f"[SIM] Progress: {sim.instr_count}/{max_steps} PC=0x{sim.pc:08x}", flush=True)
                    next_progress += progress_interval
            except HaltException as e:
                status = "halt"
                reason = e.reason
                code = e.code
                break
            except TrapException as e:
                sim.pc = sim._raise_trap(e.cause, e.tval)
            except ValueError:
                sim.pc = sim._raise_trap(2, sim.last_instr if sim.last_instr is not None else 0)
            except Exception as e:
                status = "error"
                reason = "exception"
                error = str(e)
                break
    finally:
        if stop_event is not None:
            stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1.0)

    instr_count = sim.instr_count
    signature_path = None
    if signature_begin is not None and signature_end is not None and signature_end > signature_begin:
        sig_dir = resolve_path(config.get("results_dir", "conformance/test/results")) / suite_name
        signature_path = sig_dir / (elf_path.name + ".sig")
        dump_signature(sim, signature_begin, signature_end, signature_path)

    return {
        "elf": str(elf_path),
        "status": status,
        "reason": reason,
        "code": code,
        "error": error,
        "signature": str(signature_path) if signature_path else None,
        "instr_count": instr_count,
    }


def build_riscv_tests(config, gcc_path, prefix):
    suite = config["riscv_tests"]
    isa_dir = resolve_path(suite["isa_dir"])
    groups = suite.get("groups", [])
    if not groups:
        raise RuntimeError("riscv_tests.groups is empty")
    targets = []
    for group in groups:
        makefrag = isa_dir / group / "Makefrag"
        tests = parse_makefrag_tests(makefrag, group)
        if not tests:
            raise RuntimeError(f"No tests found in {makefrag}")
        targets.extend([f"{group}-p-{name}" for name in tests])
    cmd = ["make", "-C", str(isa_dir), "XLEN=32"] + targets
    env = os.environ.copy()
    env["RISCV_GCC"] = gcc_path
    if prefix:
        env["RISCV_PREFIX"] = prefix
    env["RISCV_GCC_OPTS"] = config.get(
        "gcc_opts",
        "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -march=rv32imc_zicsr_zifencei -mabi=ilp32",
    )
    run_cmd(cmd, env=env)
    tests = []
    for group in groups:
        for path in sorted(isa_dir.glob(f"{group}-p-*")):
            if path.suffix:
                continue
            if path.is_file():
                tests.append(path)
    return tests


def build_arch_tests(config, gcc_path):
    suite = config["arch_tests"]
    suite_dir = resolve_path(suite["suite_dir"])
    env_dir = resolve_path(suite["env_dir"])
    plugin_env = resolve_path(suite["plugin_env"])
    build_dir = resolve_path(suite.get("build_dir", "conformance/test/results/riscv-arch-test/build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    groups = suite.get("groups", [])
    skip_prefixes = suite.get("skip_prefixes", [])
    reuse_build = suite.get("reuse_build", True)
    defines = suite.get("defines", [])
    if not groups:
        raise RuntimeError("arch_tests.groups is empty")

    gcc_opts = config.get(
        "gcc_opts",
        "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -march=rv32imc_zicsr_zifencei -mabi=ilp32",
    )
    opts = shlex.split(gcc_opts)
    tests = []
    for group in groups:
        src_dir = suite_dir / group / "src"
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.glob("*.S")):
            stem = src.stem
            if any(stem.startswith(prefix) for prefix in skip_prefixes):
                continue
            extra_defines = list(defines)
            with open(src, "r") as f:
                content = f.read()
            if "rvtest_mtrap_routine" in content:
                extra_defines.append("rvtest_mtrap_routine")
            if "rvtest_strap_routine" in content:
                extra_defines.append("rvtest_strap_routine")
            out = build_dir / f"{group}-{src.stem}.elf"
            if reuse_build and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
                tests.append(out)
                continue
            cmd = [
                gcc_path,
                *opts,
                "-DXLEN=32",
                "-DTEST_CASE_1",
                *[f"-D{define}" for define in extra_defines],
                f"-I{env_dir}",
                f"-I{plugin_env}",
                f"-T{plugin_env / 'link.ld'}",
                str(src),
                "-o",
                str(out),
            ]
            run_cmd(cmd)
            tests.append(out)
    return tests


def build_torture_tests(config, gcc_path):
    suite = config["torture_tests"]
    suite_dir = resolve_path(suite["suite_dir"])
    output_dir = resolve_path(suite["output_dir"])
    env_dir = resolve_path(suite["env_dir"])
    build_dir = resolve_path(suite.get("build_dir", "conformance/test/results/riscv-torture/build"))
    build_dir.mkdir(parents=True, exist_ok=True)

    should_generate = suite.get("generate", False)
    if should_generate or not list(output_dir.glob("*.S")):
        env = os.environ.copy()
        sbt_home = suite_dir / ".sbt"
        sbt_boot = sbt_home / "boot"
        ivy_home = suite_dir / ".ivy2"
        sbt_home.mkdir(parents=True, exist_ok=True)
        sbt_boot.mkdir(parents=True, exist_ok=True)
        ivy_home.mkdir(parents=True, exist_ok=True)
        sbt_launch = suite_dir / "sbt-launch.jar"
        sbt_props = (
            f"-Dsbt.global.base={sbt_home} "
            f"-Dsbt.boot.directory={sbt_boot} "
            f"-Dsbt.ivy.home={ivy_home} "
        )
        java_cmd = "java"
        java_home = suite.get("java_home") or os.environ.get("JAVA_HOME")
        if java_home:
            java_home = str(resolve_path(java_home))
            env["JAVA_HOME"] = java_home
            env["PATH"] = f"{Path(java_home) / 'bin'}:{env.get('PATH', '')}"
            java_cmd = str(Path(java_home) / "bin" / "java")
        major = java_major_version(java_cmd)
        if major is not None and major >= 9:
            # sbt 0.13 expects these properties to exist on Java 9+.
            sbt_props += "-Djava.ext.dirs= -Dsun.boot.class.path= "
        repo_cfg = suite.get("sbt_repo_config")
        if repo_cfg:
            repo_cfg_path = resolve_path(repo_cfg)
            sbt_props += (
                f"-Dsbt.override.build.repos=true "
                f"-Dsbt.repository.config={repo_cfg_path} "
            )
        sbt_props += (
            "-Dsbt.supershell=false -Dsbt.log.noformat=true -Dsbt.ci=true "
            "-Djline.terminal=jline.UnsupportedTerminal -Dscala.color=false "
        )
        env["SBT"] = f"{java_cmd} {sbt_props}-Xmx1G -Xss8M -jar {sbt_launch}"
        env["HOME"] = str(suite_dir)
        env["IVY_HOME"] = str(ivy_home)
        gen_opts = suite.get("generator_options")
        cmd = ["make", "-C", str(suite_dir), "gen"]
        if gen_opts:
            cmd.append(f"OPTIONS={gen_opts}")
        old_ttou = None
        if hasattr(signal, "SIGTTOU"):
            old_ttou = signal.getsignal(signal.SIGTTOU)
            signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        try:
            run_cmd(cmd, env=env)
        finally:
            if old_ttou is not None:
                signal.signal(signal.SIGTTOU, old_ttou)

    gcc_opts = config.get(
        "gcc_opts",
        "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -march=rv32imc_zicsr_zifencei -mabi=ilp32",
    )
    opts = shlex.split(gcc_opts)
    tests = []
    for src in sorted(output_dir.glob("*.S")):
        out = build_dir / f"{src.stem}.elf"
        cmd = [
            gcc_path,
            *opts,
            f"-I{env_dir}",
            f"-T{env_dir / 'link.ld'}",
            str(src),
            "-o",
            str(out),
        ]
        run_cmd(cmd)
        tests.append(out)
    return tests


def interpret_result(result, suite_name):
    status = result["status"]
    reason = result["reason"]
    code = result["code"]
    if status == "error":
        return "error"
    if status == "timeout":
        return "timeout"
    if suite_name == "riscv-tests":
        if reason == "exit":
            return "pass" if code == 0 else "fail"
        if reason == "tohost":
            return "pass" if code == 1 else "fail"
        return "fail"
    if suite_name in ("riscv-arch-test", "riscv-torture"):
        if reason == "tohost":
            return "pass" if code == 1 else "fail"
        return "fail"
    return "fail"


def run_suite(config_path):
    config_path = resolve_path(config_path)
    config = load_config(config_path)
    suite_name = config.get("suite")
    if not suite_name:
        raise RuntimeError("config is missing 'suite'")

    gcc_path, nm_path, prefix = toolchain_paths(config)
    if suite_name == "riscv-tests":
        tests = build_riscv_tests(config, gcc_path, prefix)
    elif suite_name == "riscv-arch-test":
        tests = build_arch_tests(config, gcc_path)
    elif suite_name == "riscv-torture":
        tests = build_torture_tests(config, gcc_path)
    else:
        raise RuntimeError(f"Unknown suite: {suite_name}")

    results = []
    total = len(tests)
    for idx, test in enumerate(tests, 1):
        print(f"[{idx}/{total}] {test.name}")
        result = run_elf(test, config, str(config_path), nm_path, suite_name)
        outcome = interpret_result(result, suite_name)
        result["outcome"] = outcome
        results.append(result)
        if "instr_count" in result:
            print(f"[SIM] Instructions: {result['instr_count']}")
        print(f"[{outcome}] {Path(result['elf']).name}")

    results_path = resolve_path(config.get(
        "results_path",
        f"conformance/test/results/{suite_name}-results.json",
    ))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    failures = [r for r in results if r["outcome"] not in ("pass",)]
    print(f"Completed {suite_name}: {len(results)} tests, {len(failures)} failures")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="Build and run RISC-V conformance suites")
    parser.add_argument("--config", required=True, help="Path to suite config JSON")
    args = parser.parse_args()
    sys.exit(run_suite(args.config))


if __name__ == "__main__":
    main()
