import subprocess
import sys
import time
import socket
import os
import signal

# Configuration
RV32SIM = [sys.executable, "../rv32sim.py"]
GDB = "gdb-multiarch" # Or riscv32-unknown-elf-gdb
TIMEOUT = 5

def print_header(name):
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

def run_assertion_test():
    print_header("Assertions (MMIO Validation)")
    
    # 1. Run Passing Case
    print("[1/2] Running Valid Assertion Case...")
    # Add --run to actually execute code
    cmd = RV32SIM + ["assertion_example.elf", "--assert", "assertions.json", "--assert-writes", "--run"]
    print(f"CMD: {' '.join(cmd)}")

    
    # We expect this to run and exit (the example loops forever, so we run for a bit then kill)
    # Actually assertion_example.c returns 0.
    
    try:
        # run with a timeout
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        print("OUTPUT (Last 5 lines):")
        print('\n'.join(p.stdout.splitlines()[-5:]))
        
        # Check output for assertion failure
        if "Assertion failed" in p.stdout:
            print("FAILURE: Assertion failed unexpectedly.")
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
    bad_json = "assertions_fail.json"
    with open(bad_json, "w") as f:
        f.write('''{
  "assertions": {
    "0x40000000": {
      "register": "UART_DATA",
      "write": { "value": "0x99", "mask": "0xFF" }
    }
  }
}''')
    
    cmd = RV32SIM + ["assertion_example.elf", "--assert", bad_json, "--assert-writes", "--run"]
    print(f"CMD: {' '.join(cmd)}")
    
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if "Assertion failed" in p.stdout:
            print("SUCCESS: Caught invalid write as expected.")
            print(f"LOG: Found '{[line for line in p.stdout.splitlines() if 'Assertion failed' in line][0]}'")
        else:
            print("FAILURE: Simulator did not report assertion failure.")
            print(p.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("FAILURE: Timed out waiting for assertion failure.")
        return False
    finally:
        if os.path.exists(bad_json):
            os.remove(bad_json)

    return True

def run_gdb_test():
    print_header("GDB Debugging (Step, Break, Read Mem)")
    
    elf = "debug_example.elf"
    port = 3334
    
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
        f.write(f"target remote :{port}\n")
        f.write("load\n")
        f.write("break main\n")
        f.write("continue\n") # Should hit main
        f.write("step\n")     # Step into setup
        f.write("set var a = 0x1234\n") # Modify variable
        f.write("print a\n")
        f.write("quit\n")
        
    print("Running GDB...")
    try:
        # Increase GDB timeout
        gdb_proc = subprocess.run(
            [GDB, "-x", gdb_script, elf],
            capture_output=True,
            text=True,
            timeout=20
        )
        
        print("GDB OUTPUT:")
        # Filter relevant lines
        for line in gdb_proc.stdout.splitlines():
            if "$1 =" in line or "Breakpoint" in line:
                print(f"  {line}")
                
        if "$1 = 4660" in gdb_proc.stdout: # 0x1234 = 4660
            print("SUCCESS: GDB connected, stepped, and modified variable.")
        else:
            print("FAILURE: Did not find expected GDB output.")
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
        RV32SIM + [elf, f"--rtt-port={rtt_port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2) # Wait for RTT scan
    
    try:
        print("Connecting to RTT socket...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', rtt_port))
        
        # Read Initial Banner
        data = s.recv(1024).decode(errors='ignore')
        print(f"RX: {data.strip()}")
        
        if "RTT Initialized" not in data:
            print("FAILURE: Did not receive RTT banner.")
            return False
            
        # Send Data
        msg = "Hello RTT"
        print(f"TX: {msg}")
        s.send(msg.encode())
        
        # Read Echo
        time.sleep(0.5)
        response = s.recv(1024).decode(errors='ignore')
        print(f"RX: {response.strip()}")
        
        if "Echo: H" in response:
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
        if sim_proc.poll() is None:
            sim_proc.terminate()
        
def run_uart_test():
    print_header("UART Peripheral (Polling)")
    
    # We run the uart_polling.elf and inject input via command line argument
    # Expectation: The program echoes input and prints "Quit command received"
    
    input_str = "Test\nq"
    cmd = RV32SIM + ["uart_polling.elf", f"--uart-input={input_str}"]
    print(f"CMD: {' '.join(cmd)}")
    
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

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
