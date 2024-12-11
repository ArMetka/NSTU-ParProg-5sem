CC=mpicc
FLAGS=-std=c11 -DSERVER
LFLAGS=
MODULES_C=solution.c
MODULES_O=$(MODULES_C:.c=.o)
SOLUTION_EXE=solution

all: clean
	make solution
	mpirun -use-hwthread-cpus ./$(SOLUTION_EXE)

$(SOLUTION_EXE): $(MODULES_O)
	$(CC) $(FLAGS) -o $@ $^ $(LFLAGS)
	make clean_o

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@ $(LFLAGS)

clean: clean_o clean_exe

clean_o:
	rm -rf *.o

clean_exe:
	rm -rf $(SOLUTION_EXE)