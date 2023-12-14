#################################
# RISCV Toolchain
#################################

PREFIX = riscv64-unknown-elf-

GCC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

K_VAL :=
TS :=
GEMMINI_OP := DONTCARE

#################################
# Flags
#################################

# SoC Settings
ARCH = rv64imafdc
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -O2 -fno-common -fno-builtin-printf -I../../lib/riscv_gemmini/fp32 -I../../src
CFLAGS += $(ARCHFLAGS)
CFLAGS += -DPLATFORM_RV=1 -DBM=1 -DK_VAL=$(K_VAL) -DTS=$(TS) -D$(GEMMINI_OP)
LDFLAGS = -static

include libgloss.mk

PROGRAMS = test 


.DEFAULT_GOAL := default


#################################
# Build
#################################
vpath %.c ../../src

%.o: %.c 
	$(GCC) $(CFLAGS) -c $< -o $@ -lm

%.riscv: %.o $(libgloss)
	$(GCC) $(LDFLAGS) $< -o $@ -lm

%.dump: %.riscv
	$(OBJDUMP) -D $< > $@


#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump
	$(if $(libgloss),rm -rf $(libgloss_builddir)/)

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS)) 

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
