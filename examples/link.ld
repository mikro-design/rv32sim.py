MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 128K
  SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 16M
}

SECTIONS
{
  .text : {
    *(.text.start)
    *(.text*)
    *(.rodata*)
  } > FLASH

  .data : {
    *(.data*)
    *(.sdata*)
  } > SRAM

  .bss : {
    *(.bss*)
    *(.sbss*)
  } > SRAM

  /* Stack top at end of SRAM */
  _stack_top = ORIGIN(SRAM) + LENGTH(SRAM);
}
