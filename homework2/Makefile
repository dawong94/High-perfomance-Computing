# compiler settings
CC=gcc
CFLAGS=-c -Wall -fPIC
LDFLAGS=-lm --shared

# target
LIBTARGET=libhomework2.so

# directories
SRC=src

LIB=lib
INCLUDE=include

# sources
LIBSOURCES=$(SRC)/linalg.c $(SRC)/solvers.c
LIBOBJECTS=$(LIBSOURCES:.c=.o)

.PHONY: all lib clean

default: all

all: lib

lib: $(LIBTARGET)

test: lib
	python test_homework2.py

$(LIBTARGET): $(LIBOBJECTS)
	$(CC) $(LDFLAGS) $(LIBOBJECTS) -o $(LIB)/$(LIBTARGET)

%.o : %.c
	$(CC) -I$(INCLUDE) $(CFLAGS) $< -o $@

clean:
	@rm -fr $(SRC)/*.o

clobber: clean
	@rm -fd $(LIB)/*.so
