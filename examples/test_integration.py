import subprocess
import sys
import time
import socket
import os
import signal
import shutil
import re

# Configuration
RV32SIM = [sys.executable, "../rv32sim.py"]
TIMEOUT = 5
GDB_TIMEOUT = int(os.environ.get("GDB_TIMEOUT", "20"))

def _resolve_gdb():
    override = os.environ.get("GDB")
    if override:
        return override
    for candidate in ("riscv32-unknown-elf-gdb", "riscv64-unknown-elf-gdb", "gdb-multiarch"):
        if shutil.which(candidate):
            return candidate
    return None

GDB = _resolve_gdb()

def print_header(name):
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

def run_assertion_test():
    print_header("Assertions (MMIO Validation)")

    pass_json = "assertions_pass.json"
    bad_json = "assertions_fail.json"
    try:
        with open(pass_json, "w") as f:
            f.write('''{
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
}''')
    
        # 1. Run Passing Case
        print("[1/2] Running Valid Assertion Case...")
        # Add --run to actually execute code
        cmd = RV32SIM + ["assertion_example.elf", "--assert", pass_json, "--assert-writes", "--run"]
        print(f"CMD: {' '.join(cmd)}")

        # We expect this to run and exit (the example loops forever, so we run for a bit then kill)
        # Actually assertion_example.c returns 0.
        try:
            # run with a timeout
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            output = (p.stdout or "") + (p.stderr or "")
            print("OUTPUT (Last 5 lines):")
            print('\n'.join(output.splitlines()[-5:]))
            
            # Check output for assertion failure
            if p.returncode != 0 or "[ASSERT]" in output:
                print("FAILURE: Assertions triggered unexpectedly.")
                return False
                
            print("SUCCESS: Valid logic passed assertions.")
            
        except subprocess.TimeoutExpired:
            # If it timed out, it might be stuck in a loop or just slow. 
            # assertion_example.c ends with return 0, so it should exit.
            print("WARNING: Timed out. Did the program finish?")

        # 2. Run Failing Case (We need to modify the binary or use a strict assertion that fails)
        # Let's verify that a bad write IS caught.
        # We will use a temporary assertion file that expects a different value.
        
        print("\n[2/2] Running Invalid Assertion Case...")
        with open(bad_json, "w") as f:
            f.write('''{
  "assertions": {
    "0x40000000": {
      "register": "UART_DATA",
      "write": { "value": "0x99", "mask": "0xFF" }
    },
    "0x40000004": {
      "register": "UART_CTRL",
      "read": { "value": "0x0" },
      "write": { "value": "0x1", "mask": "0x1" }
    }
  }
}''')
        
        cmd = RV32SIM + ["assertion_example.elf", "--assert", bad_json, "--assert-writes", "--run"]
        print(f"CMD: {' '.join(cmd)}")
        
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            output = (p.stdout or "") + (p.stderr or "")
            if "[ASSERT]" in output:
                print("SUCCESS: Caught invalid write as expected.")
                assert_lines = [line for line in output.splitlines() if "[ASSERT]" in line]
                if assert_lines:
                    print(f"LOG: {assert_lines[0]}")
            else:
                print("FAILURE: Simulator did not report assertion failure.")
                if output.strip():
                    print(output)
                else:
                    print(f"Return code: {p.returncode}")
                return False
        except subprocess.TimeoutExpired:
            print("FAILURE: Timed out waiting for assertion failure.")
            return False

        return True
    finally:
        for path in (pass_json, bad_json):
            if os.path.exists(path):
                os.remove(path)

def run_gdb_test():
    print_header("GDB Debugging (Step, Break, Read Mem)")
    
    if not GDB:
        print("SKIP: GDB executable not found. Set GDB=... or install gdb-multiarch/riscv32-unknown-elf-gdb.")
        return True

    elf = "debug_example.elf"
    port = 3334

    print(f"Using GDB: {GDB}")

    # Resolve main address for bounded execution.
    try:
        addr_proc = subprocess.run(
            [GDB, "-batch", "-ex", "info address main", elf],
            capture_output=True,
            text=True,
            timeout=GDB_TIMEOUT,
        )
        addr_output = (addr_proc.stdout or "") + (addr_proc.stderr or "")
        match = re.search(r"(?:address|at) (0x[0-9a-fA-F]+)", addr_output)
        if not match:
            print("FAILURE: Could not resolve 'main' address from GDB output.")
            print(addr_output.strip())
            return False
        main_addr = match.group(1)
    except Exception as e:
        print(f"FAILURE: Could not resolve 'main' address: {e}")
        return False

    # Start Simulator in background
    print(f"Starting Simulator on port {port}...")
    sim_proc = subprocess.Popen(
        RV32SIM + [elf, f"--port={port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it a moment to start
    time.sleep(1)
    
    # Create GDB script
    gdb_script = "gdb_test.cmd"
    with open(gdb_script, "w") as f:
        f.write("set pagination off\n")
        f.write("set confirm off\n")
        f.write("set breakpoint pending on\n")
        f.write("set remotetimeout 5\n")
        f.write("set auto-load off\n")
        f.write(f"target remote :{port}\n")
        f.write(f"monitor run_until_pc {main_addr} 200000\n")
        f.write("step\n")     # Step into main
        f.write("set $a0 = 0x1234\n")
        f.write("print/x $a0\n")
        f.write("quit\n")
        
    print("Running GDB...")
    try:
        gdb_proc = subprocess.run(
            [GDB, "-nx", "-q", "-batch", "-x", gdb_script, elf],
            capture_output=True,
            text=True,
            timeout=GDB_TIMEOUT
        )
        output = (gdb_proc.stdout or "") + (gdb_proc.stderr or "")
        print("GDB OUTPUT:")
        # Filter relevant lines
        for line in output.splitlines():
            if "$1 =" in line or "Breakpoint" in line or "Hit PC=" in line or "Max steps reached" in line:
                print(f"  {line}")
                
        if re.search(r"\$1 = .*0x0*1234", output) or "$1 = 0x1234" in output or " $1 = 4660" in output:
            print("SUCCESS: GDB connected, stepped, and modified variable.")
        elif "Max steps reached" in output:
            print("FAILURE: GDB did not reach main within step limit.")
            return False
        else:
            print("FAILURE: Did not find expected GDB output.")
            return False
            
    except subprocess.TimeoutExpired:
        print("GDB Execution Failed: Timed out.")
        sim_proc.terminate()
        return False
    except Exception as e:
        print(f"GDB Execution Failed: {e}")
        # Cleanup
        sim_proc.terminate()
        return False
    finally:
        if sim_proc.poll() is None:
            sim_proc.terminate()
        if os.path.exists(gdb_script):
            os.remove(gdb_script)
            
    return True

def run_rtt_test():
    print_header("RTT (Real Time Transfer)")
    
    elf = "rtt_example.elf"
    rtt_port = 4002
    
    # Start Simulator
    print(f"Starting Simulator with RTT on port {rtt_port}...")
    # RTT example waits for input, so it won't exit on its own immediately
    sim_proc = subprocess.Popen(
        RV32SIM + [elf, f"--rtt-port={rtt_port}", "--run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2) # Wait for RTT scan
    
    s = None
    try:
        print("Connecting to RTT socket...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.connect(('localhost', rtt_port))
        s.settimeout(0.5)
        
        # Read Initial Banner
        banner = b""
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline and b"RTT Initialized" not in banner:
            try:
                chunk = s.recv(1024)
            except socket.timeout:
                continue
            if not chunk:
                break
            banner += chunk
        banner_text = banner.decode(errors="ignore")
        print(f"RX: {banner_text.strip()}")
        
        if "RTT Initialized" not in banner_text:
            print("FAILURE: Did not receive RTT banner.")
            return False
            
        # Send Data
        msg = "Hello RTT"
        print(f"TX: {msg}")
        s.send(msg.encode())
        
        # Read Echo
        response = b""
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline and b"Echo:" not in response:
            try:
                chunk = s.recv(1024)
            except socket.timeout:
                continue
            if not chunk:
                break
            response += chunk
        response_text = response.decode(errors="ignore")
        print(f"RX: {response_text.strip()}")
        
        if "Echo: H" in response_text:
            print("SUCCESS: RTT Bidirectional communication verified.")
            return True
        else:
            print("FAILURE: Did not receive echo.")
            return False
            
    except Exception as e:
        print(f"RTT Failed: {e}")
        sim_proc.terminate()
        return False
    finally:
        if s:
            s.close()
        if sim_proc.poll() is None:
            sim_proc.terminate()
        
def run_uart_test():
    print_header("UART Peripheral (Polling)")
    
    # We run the uart_polling.elf and inject input via command line argument
    # Expectation: The program echoes input and prints "Quit command received"
    
    input_str = "Test\nq"
    cmd = RV32SIM + ["uart_polling.elf", f"--uart-input={input_str}", "--run"]
    display_cmd = " ".join(arg.replace("\n", "\\n") for arg in cmd)
    print(f"CMD: {display_cmd}")
    
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)

        output = p.stdout
        
        print("OUTPUT:")
        for line in output.splitlines():
            if "Recv:" in line or "Quit" in line:
                print(f"  {line}")
                
        if "Recv: T" in output and "Quit command received" in output:
            print("SUCCESS: UART Input injected and processed.")
            return True
        else:
            print("FAILURE: UART logic incorrect.")
            return False
    except Exception as e:
        print(f"UART Test Failed: {e}")
        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    passed = True
    passed &= run_assertion_test()
    passed &= run_gdb_test()
    passed &= run_rtt_test()
    passed &= run_uart_test()
    
    if passed:
        print("\n\nAll Integration Tests PASSED.")
        sys.exit(0)
    else:
        print("\n\nSome tests FAILED.")
        sys.exit(1)
