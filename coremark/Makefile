CROSS_COMPILE ?= riscv32-unknown-elf-
CC := $(CROSS_COMPILE)gcc
SIZE := $(CROSS_COMPILE)size

ARCH ?= rv32im
ABI ?= ilp32

CONFIG_OPTIMIZE_LEVEL ?= 2
CONFIG_OPTIMIZATION ?=
EXTRA_CFLAGS ?=

ITERATIONS ?= 1
TOTAL_DATA_SIZE ?= 200
CPU_FREQ_HZ ?= 1000000

BUILD_DIR ?= build
ELF := $(BUILD_DIR)/coremark.elf
MAP := $(BUILD_DIR)/coremark.map

LDSCRIPT ?= linker.ld

SRCS := core_list_join.c core_main.c core_matrix.c core_state.c core_util.c \
	core_portme.c ee_printf.c
ASM_SRCS := startup.S
OBJS := $(addprefix $(BUILD_DIR)/,$(SRCS:.c=.o) $(ASM_SRCS:.S=.o))

CPPFLAGS := -I. -DITERATIONS=$(ITERATIONS) -DTOTAL_DATA_SIZE=$(TOTAL_DATA_SIZE) \
	-DCPU_FREQ_HZ=$(CPU_FREQ_HZ)
BASE_CFLAGS := -march=$(ARCH) -mabi=$(ABI) -ffreestanding -fno-builtin \
	-Wall -Wextra -Wno-unused-parameter
OPT_FLAGS := -O$(CONFIG_OPTIMIZE_LEVEL)
CFLAGS := $(BASE_CFLAGS) $(OPT_FLAGS) $(CONFIG_OPTIMIZATION) $(EXTRA_CFLAGS) -std=c99

LDFLAGS := $(BASE_CFLAGS) $(OPT_FLAGS) $(CONFIG_OPTIMIZATION) $(EXTRA_CFLAGS) \
	-nostdlib -Wl,--gc-sections -Wl,-Map=$(MAP) -T $(LDSCRIPT)
LDLIBS := -lgcc

all: $(ELF)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.S | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(LDSCRIPT)
	$(CC) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@
	$(SIZE) $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
