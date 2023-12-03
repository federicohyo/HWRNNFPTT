CC=gcc
LIBS=-lm
CFLAGS=-I$(IDIR) -Wall -D MNIST_ONLINE_BY_PIXELS #MNIST_OFFLINE_BY_PIXELS #BY_PIXEL_FPTT_K_56


SDIR=./src
IDIR=./include
ODIR=./obj
LDIR=./lib


_DEPS = headers.h matrix_ops.h lstm.h caselist.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = matrix_ops.o lstm.o caselist.o main.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

run: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ ./run 
