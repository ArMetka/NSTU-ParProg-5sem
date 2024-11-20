CC1=gcc
CC2=nvcc
FLAGS1=-std=c99
FLAGS2=
MODULES_CU=solution.cu stack.cu
MODULES_O=$(MODULES_CU:.cu=.o)
GENERATOR_MODULES_C=array_generator.c
GENERATOR_MODULES_O=$(GENERATOR_MODULES_C:.c=.o)
CFLAGS=-rdc=true -lcudadevrt
LFLAGS=-rdc=true -lcudadevrt
SOLUTION_EXE=solution
GENERATOR_EXE=generator
INPUT_FILE=unsorted.txt
OUTPUT_FILE_SEQ=sorted_seq.txt
OUTPUT_FILE_PAR=sorted_par.txt

all: clean generator
	make solution

$(SOLUTION_EXE): $(MODULES_O)
	$(CC2) -o $@ $^ $(LFLAGS)
	make clean_o
	./$(SOLUTION_EXE) $(INPUT_FILE)

recursive: clean generator
	$(CC2) $(FLAGS2) $(CFLAGS) -DRECURSIVE -c solution.cu -o solution.o
	$(CC2) $(FLAGS2) $(CFLAGS) -c stack.cu -o stack.o
	$(CC2) -o $(SOLUTION_EXE) solution.o stack.o $(LFLAGS)
	./$(SOLUTION_EXE) $(INPUT_FILE)

generator: $(GENERATOR_MODULES_O)
	$(CC1) -o $(GENERATOR_EXE) $(GENERATOR_MODULES_O)
	make clean_o
	./$(GENERATOR_EXE) $(INPUT_FILE)

%.o: %.c
	$(CC1) $(FLAGS1) -c $< -o $@

%.o: %.cu
	$(CC2) $(FLAGS2) $(CFLAGS) -c $< -o $@

clean: clean_o clean_io clean_exe

clean_o:
	rm -rf *.o

clean_io:
	rm -rf $(INPUT_FILE)
	rm -rf $(OUTPUT_FILE_SEQ)
	rm -rf $(OUTPUT_FILE_PAR)

clean_exe:
	rm -rf $(SOLUTION_EXE)
	rm -rf $(GENERATOR_EXE)