import subprocess
import time
import sys
import os
import signal

_ASSERTIONS_JSON = '''{
  "assertions": {
    "0x40000000": {
      "register": "UART_DATA",
      "write": { "value": "0x41", "mask": "0xFF" }
    },
    "0x40000004": {
      "register": "UART_CTRL",
      "read": { "value": "0x0" },
      "write": { "value": "0x1", "mask": "0x1" }
    }
  }
}'''


def _ensure_assertions_file(path):
    if os.path.exists(path):
        return False
    with open(path, "w") as f:
        f.write(_ASSERTIONS_JSON)
    return True


def run_gdb_load_demo(elf_name, sim_args=[], gdb_cmds=[]):
    print(f"\n{'='*60}")
    print(f"DEMO: Loading {elf_name} via GDB")
    print(f"{'='*60}")

    port = 3335
    
    # 1. Start Simulator (Empty, waiting for GDB)
    # Note: We do NOT pass the ELF here.
    sim_cmd = [sys.executable, "../rv32sim.py", f"--port={port}"] + sim_args
    print(f"TERMINAL 1: {' '.join(sim_cmd)}")
    
    sim_proc = subprocess.Popen(
        sim_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid # Create new process group for clean killing
    )
    
    # Wait for sim to initialize
    time.sleep(1)
    
    # 2. Prepare GDB Commands
    # We use a temporary GDB script to ensure strict ordering
    gdb_script_content = f"""
target remote localhost:{port}
load
break main
""" 
    for cmd in gdb_cmds:
        gdb_script_content += f"{cmd}\n"
    
    gdb_script_content += "quit\n"
    
    with open("demo.gdb", "w") as f:
        f.write(gdb_script_content)
        
    # 3. Run GDB
    gdb_path = "riscv32-unknown-elf-gdb"
    gdb_cmd = [gdb_path, elf_name, "-x", "demo.gdb"]
    print(f"TERMINAL 2: {' '.join(gdb_cmd)}")
    
    try:
        result = subprocess.run(
            gdb_cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print("\n--- GDB OUTPUT ---")
        # Filter relevant output to show it worked
        for line in result.stdout.splitlines():
            if "Transfer rate" in line:
                print(line)
            if "Breakpoint" in line:
                print(line)
            if "value" in line or "=" in line:
                print(line)
                
        if result.returncode == 0:
            print("\nSTATUS: GDB Load Successful")
        else:
            print(f"\nSTATUS: GDB Failed (Return Code {result.returncode})")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("STATUS: GDB Timed Out")
    finally:
        # Cleanup
        os.killpg(os.getpgid(sim_proc.pid), signal.SIGTERM)
        if os.path.exists("demo.gdb"):
            os.remove("demo.gdb")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    assertions_path = "assertions_demo.json"
    created = _ensure_assertions_file(assertions_path)
    try:
        # Demo 1: Simple Debug
        run_gdb_load_demo("debug_example.elf", 
                          gdb_cmds=["continue", "print a", "print b"])

        # Demo 2: Assertions (Sim must be configured with --assert before load)
        run_gdb_load_demo("assertion_example.elf", 
                          sim_args=["--assert", assertions_path, "--assert-writes"],
                          gdb_cmds=["continue"]) # logic runs and exits
    finally:
        if created and os.path.exists(assertions_path):
            os.remove(assertions_path)
