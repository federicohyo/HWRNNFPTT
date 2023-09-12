CC=gcc
CFLAGS=-I$(IDIR) -Wall
LIBS=-lm


SDIR=./src
IDIR=./include
ODIR=./obj
LDIR=./lib


_DEPS = headers.h lstm.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = lstm.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

run: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
