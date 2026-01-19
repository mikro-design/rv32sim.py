CC = riscv32-unknown-elf-gcc
CFLAGS = -march=rv32imc_zicsr -mabi=ilp32 -nostdlib -T link.ld -O2 -g -fno-builtin

all: debug_example.elf assertion_example.elf rtt_example.elf matrix_example.elf csr_example.elf uart_polling.elf

uart_polling.elf: uart_polling.c crt0.s link.ld
	$(CC) $(CFLAGS) -o $@ crt0.s uart_polling.c

csr_example.elf: csr_example.c crt0.s link.ld
	$(CC) $(CFLAGS) -o $@ crt0.s csr_example.c


debug_example.elf: debug_example.c crt0.s link.ld
	$(CC) $(CFLAGS) -o $@ crt0.s debug_example.c

assertion_example.elf: assertion_example.c crt0.s link.ld
	$(CC) $(CFLAGS) -o $@ crt0.s assertion_example.c

rtt_example.elf: rtt_example.c crt0.s link.ld
	$(CC) $(CFLAGS) -o $@ crt0.s rtt_example.c

matrix_example.elf: matrix_example.c crt0.s link.ld
	$(CC) $(CFLAGS) -o $@ crt0.s matrix_example.c

test: all
	python3 test_examples.py
	python3 test_integration.py

clean:
	rm -f *.elf *.o
