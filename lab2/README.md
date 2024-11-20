# Лабораторная работа №2
### Методы параллельного программирования для графического процессора в среде NVidia CUDA.

### Задача №55
Быстрая сортировка

### Решение:

#### Всё и сразу

```bash
make
```

#### Генератор массивов случайных чисел

```bash
make generator
```

Исполняемый файл `generator`, вывод в файл по умолчанию `unsorted.txt`\
Ручной запуск: `./generator [FILENAME]`\
`FILENAME` -> имя файла для вывода сгенерированного массива

#### Решение задачи

- Итеративный алгоритм
```bash
make solution
```

- Рекурсивный алгоритм (Compute Capability 3.5+)
```bash
make recursive
```

Исполняемый файл `solution`, файл на вход по умолчанию `unsorted.txt`\
Ручной запуск: `./solution [FILENAME]`\
`FILENAME` -> имя файла с входным массивом

#### Очистка директории
`make clean_o` -> удалить объектные файлы\
`make clean_io` -> удалить файлы ввода/вывода (unsorted.txt, sorted_seq.txt, sorted_par.txt)\
`make clean_exe` -> удалить исполняемые файлы (solution и generator)\
`make clean` -> clean_o && clean_io && clean_exe

#### Исходные файлы
`solution.cu, solution.cuh` -> файлы, содержащие решение задачи\
`stack.cu, stack.cuh` -> файлы, содержащие реализацию стека\
`array_generator.c` -> генератор исходных данных (случайных массивов)\
`Makefile` -> файл для автоматической сборки программы
