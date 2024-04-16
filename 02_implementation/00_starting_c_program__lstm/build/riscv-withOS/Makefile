prefix      := /usr/local
abs_top_srcdir  :=  /home/yicheng/Desktop/ms_prj/sfptt_c/02_implementation/00_starting_c_program__lstm/build/..
XLEN            := 64
RISCVTOOLS      := @RISCVTOOLS@
ROCC = examples
RUNNER         := "spike --extension=gemmini "

include $(abs_top_srcdir)/build/riscv-withOS/Makefrag

tests = \
	test \

K_VAL :=
TS :=
DTYPE := 
GEMMINI_OP := DONTCARE
RVMC_OP := DONTCARE

tests_linux = $(tests:=-linux)
tests_pk = $(tests:=-pk)

lib_dir = $(abs_top_srcdir)/lib/riscv_gemmini/$(DTYPE)
BENCH_COMMON = /home/yicheng/Desktop/ms_prj/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(lib_dir)/include/gemmini.h $(lib_dir)/include/gemmini_params.h $(lib_dir)/include/gemmini_testutils.h $(src_dir)/headers.h $(src_dir)/counters_timers.h $(src_dir)/network_params_riscv.h $(src_dir)/matrix_ops.h $(src_dir)/lstm.h $(src_dir)/caselist.h 


CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
	-Wno-incompatible-pointer-types \
	-I$(src_dir) \
    -I$(lib_dir) \
    -I$(lib_dir)/include \
    -I$(src_dir)/../data/c_headers \
    -DPLATFORM_RV=1 \
    -DK_VAL=$(K_VAL) \
    -DTS=$(TS) \
    -D$(DTYPE) \
    -D$(GEMMINI_OP) \
    -D$(RVMC_OP) \
    -DNCORES=$(NCORES) \

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
	-DBAREMETAL=1 \


all: $(tests_linux) $(tests_pk)

vpath %.c $(src_dir)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -o $@ -lm

junk += $(tests_linux) $(tests_pk)

