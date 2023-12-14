CC=gcc
LIBS=-lm
CFLAGS=-I$(IDIR) -Wall -D PLATFORM_X86


SDIR=../../src
IDIR=../../src
ODIR=./obj


_DEPS = headers.h counters_timers.h network_params_x86.h matrix_ops.h lstm.h caselist.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = test.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

bin: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ ./bin
