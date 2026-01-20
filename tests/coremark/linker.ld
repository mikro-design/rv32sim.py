ENTRY(_start)

MEMORY
{
	FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 128K
	RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 16M
}

SECTIONS
{
	.text :
	{
		KEEP(*(.text.startup))
		*(.text .text.*)
		*(.rodata .rodata.*)
	} > FLASH

	.data : AT (ADDR(.text) + SIZEOF(.text))
	{
		. = ALIGN(4);
		__data_start = .;
		*(.sdata .sdata.*)
		*(.data .data.*)
		. = ALIGN(4);
		__data_end = .;
	} > RAM
	__data_load_start = LOADADDR(.data);

	.bss (NOLOAD) :
	{
		. = ALIGN(4);
		__bss_start = .;
		*(.sbss .sbss.*)
		*(.bss .bss.*)
		*(COMMON)
		. = ALIGN(4);
		__bss_end = .;
	} > RAM

	__global_pointer$ = __data_start + 0x800;
	__heap_start = __bss_end;
	__stack_top = ORIGIN(RAM) + LENGTH(RAM);
	__heap_end = __stack_top;
}
