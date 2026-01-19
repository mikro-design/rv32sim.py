import sys
import os
import time

# Add parent directory to path to import rv32sim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rv32sim import RV32Sim, HaltException
from simple_uart import SimpleUART

def run_test(elf_path, timeout=5, uart_input=None):
    print("-" * 60)
    print(f"TEST: {os.path.basename(elf_path)}")
    print(f"PATH: {elf_path}")
    
    sim = RV32Sim()
    
    # Enable UART if input provided
    uart = SimpleUART(sim)
    if uart_input:
        display_input = uart_input.replace('\n', '\\n')
        print(f"INPUT: Queuing UART data: '{display_input}'")
        uart.queue_input(uart_input)
        
    try:
        sim.load_elf(elf_path)
        # Initialize SP
        stack_top = sim.get_stack_top()
        sim.regs[2] = (stack_top - 16) & 0xffffffff
        print(f"SETUP: SP initialized to 0x{sim.regs[2]:08x}")
    except Exception as e:
        print(f"FAILED to load: {e}")
        return False

    print("EXEC: Running simulation...")
    print("-" * 20 + " Program Output " + "-" * 20)
    start_time = time.time()
    steps = 0
    
    try:
        while True:
            sim.execute()
            steps += 1
            if steps % 5000 == 0:
                if time.time() - start_time > timeout:
                    print(f"\nTIMEOUT after {steps} steps")
                    return False
    except HaltException as e:
        duration = time.time() - start_time
        print("-" * 56)
        print(f"HALT: {e.reason} (Code: {e.code})")
        print(f"STATS: {steps} instructions in {duration:.4f}s ({(steps/duration)/1000:.2f} KIPS)")
        
        if e.reason == "exit" and e.code == 0:
            print("RESULT: PASSED")
            return True
        else:
            print(f"RESULT: FAILED (code {e.code})")
            return False
    except Exception as e:
        print(f"\nCRASHED: {e}")
        return False

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    tests = [
        (os.path.join(base_dir, "csr_example.elf"), None),
        (os.path.join(base_dir, "matrix_example.elf"), None),
        (os.path.join(base_dir, "uart_polling.elf"), "TestMessage\nq")
    ]
    
    failed = []
    for test, u_input in tests:
        if not os.path.exists(test):
            print(f"Test not found: {test}")
            failed.append(test)
            continue
            
        if not run_test(test, uart_input=u_input):
            failed.append(test)
            
    if failed:
        print(f"\nFailed tests: {failed}")
        sys.exit(1)
    else:
        print("\nAll automated tests passed!")
        sys.exit(0)
