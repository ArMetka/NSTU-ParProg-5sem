CC=gcc
FLAGS=-std=c11
LFLAGS=-I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/ -lOpenCL
MODULES_C=solution.c
MODULES_O=$(MODULES_C:.c=.o)
SOLUTION_EXE=solution

all: clean
	make solution

$(SOLUTION_EXE): $(MODULES_O)
	$(CC) $(FLAGS) -o $@ $^ $(LFLAGS)
	make clean_o
	./$(SOLUTION_EXE)

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@ $(LFLAGS)

clean: clean_o clean_exe

clean_o:
	rm -rf *.o

clean_exe:
	rm -rf $(SOLUTION_EXE)