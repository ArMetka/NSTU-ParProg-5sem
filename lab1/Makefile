CC=gcc
FLAGS=-Wall -Werror -Wextra -std=c99
MODULES_C=solution.c
MODULES_O=$(MODULES_C:.c=.o)
GENERATOR_MODULES_C=sparse_matrix_generator.c
GENERATOR_MODULES_O=$(GENERATOR_MODULES_C:.c=.o)
CFLAGS=-fopenmp
LFLAGS=-fopenmp
SOLUTION_EXE=solution
GENERATOR_EXE=generator
INPUT_FILE=input.txt
OUTPUT_FILE=output.txt

all: clean generator
	make solution

solution: $(MODULES_O)
	$(CC) -o $(SOLUTION_EXE) $(MODULES_O) $(LFLAGS)
	make clean_o
	./$(SOLUTION_EXE) $(INPUT_FILE)

generator: $(GENERATOR_MODULES_O)
	$(CC) -o $(GENERATOR_EXE) $(GENERATOR_MODULES_O)
	make clean_o
	./$(GENERATOR_EXE) $(INPUT_FILE)

%.o: %.c
	$(CC) $(FLAGS) $(CFLAGS) -c $< -o $@

clean: clean_o clean_io clean_exe

clean_o:
	rm -rf *.o

clean_io:
	rm -rf $(INPUT_FILE)
	rm -rf $(OUTPUT_FILE)

clean_exe:
	rm -rf $(SOLUTION_EXE)
	rm -rf $(GENERATOR_EXE)
