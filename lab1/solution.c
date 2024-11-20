#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#include "./solution.h"

/*
 * ./solution [FILENAME]
 * Find sparse matrix of size [target_row] x [target_col]
 * sparse matrix -> at least one row/column must be all zeroes (but not the first/last row/column)
 */
int main(int argc, char **argv) {
    // Return value of main
    int result = 0;

    // Variables for source matrix dimensions
    int src_row = 0;
    int src_col = 0;

    // Variables for target sparse matrix dimensions
    int target_row = 0;
    int target_col = 0;

    // Variables for target sparse matrix offset
    int offset_row1 = 0;
    int offset_row2 = 0;
    int offset_col1 = 0;
    int offset_col2 = 0;

    // Variable for file handle
    FILE *file;

    // Structs for time measurement
    struct timespec start, finish, delta1, delta2; 

    // Source, target and sum matrix
    int64 **src_matrix;
    int64 **target_matrix1;
    int64 **target_matrix2;
    int64 **sum_matrix1;
    int64 **sum_matrix2;

    // Open input file for reading
    if (open_file(argc, argv, &file)) {
        // in case of any error exit
        return 1;
    }

    // Get source matrix from file
    get_matrix(file, &src_matrix, &src_row, &src_col);

    // Close file
    fclose(file);

    // Get target matrix dimensions
    if (get_target_matrix_dimensions(&target_row, &target_col)) {
        // in case of any error exit
        return 1;
    } else if (!is_target_dimensions_valid(src_row, src_col, target_row, target_col)) {
        // in case of invalid target matrix dimensions exit
        return 1;
    }

    allocate_matrix(&target_matrix1, target_row, target_col);
    allocate_matrix(&target_matrix2, target_row, target_col);
    allocate_matrix(&sum_matrix1, src_row - target_row + 1, src_col - target_col + 1);
    allocate_matrix(&sum_matrix2, src_row - target_row + 1, src_col - target_col + 1);

    clock_gettime(CLOCK_REALTIME, &start);
    int64 matrix_sum1 = find_sparse_matrix_seq(src_matrix, target_matrix1, sum_matrix1, src_row, src_col, target_row, target_col, &offset_row1, &offset_col1);
    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta1);

    clock_gettime(CLOCK_REALTIME, &start);
    int64 matrix_sum2 = find_sparse_matrix_par(src_matrix, target_matrix2, sum_matrix2, src_row, src_col, target_row, target_col, &offset_row2, &offset_col2);
    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta2);

    if (!matrix_sum1 && !matrix_sum2) {
        // no matrix found
        printf("No sparse matrix found\n");
        result = 1;
    } else if (src_row > 10 || src_col > 10) {
        // file output
        printf("\nDone! [FILE OUTPUT]\n");
        FILE *file = fopen("output.txt", "w");
        output_final_data_file(sum_matrix1, target_matrix1, src_row, src_col, target_row, target_col, offset_row1, offset_col1,
                               matrix_sum1, delta1, sum_matrix2, target_matrix2, offset_row2, offset_col2, matrix_sum2, delta2, file);
        fclose(file);
    } else {
        // stdout output
        output_final_data(sum_matrix1, target_matrix1, src_row, src_col, target_row, target_col, offset_row1, offset_col1,
                          matrix_sum1, delta1, sum_matrix2, target_matrix2, offset_row2, offset_col2, matrix_sum2, delta2);
    }

    free_matrix(&src_matrix, src_row);
    free_matrix(&target_matrix1, target_row);
    free_matrix(&target_matrix2, target_row);
    free_matrix(&sum_matrix1, src_row - target_row + 1);
    free_matrix(&sum_matrix2, src_row - target_row + 1);

    return result;
}

int open_file(int argc, char **argv, FILE **file) {
    if (argc == 1) {
        printf("No input file specified!\n");
        return 1;
    } else if (argc > 2) {
        printf("Too many arguments!\n");
        return 1;
    } else {
        char filename[256];
        for(int count = 0; *(*(argv + 1) + count) != '\0'; count++) {
            filename[count] = *(*(argv + 1) + count);
            filename[count + 1] = '\0';
        }
        *file = fopen(filename, "r");
        if (!(*file)) {
            printf("File does not exist!\n");
            return 1;
        }
    }

    return 0;
}

void get_matrix(FILE *file, int64 ***matrix, int *rows, int *columns) {
    fscanf(file, "%d %d\n", rows, columns);
    allocate_matrix(matrix, *rows, *columns);
    
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *columns; j++) {
            int64 number;
            char c;
            fscanf(file, "%llu%c", &number, &c);
            (*matrix)[i][j] = number;
        }
    }
}

void allocate_matrix(int64 ***matrix, int rows, int columns) {
    *matrix = (int64 **)malloc(rows * sizeof(int64 *));
    for (int i = 0; i < rows; i++) {
        *(*matrix + i) = (int64 *)malloc(columns * sizeof(int64));
    }
}

void free_matrix(int64 ***matrix, int rows) {
    for(int i = 0; i < rows; i++) {
        free(*(*matrix + i));
    }
    free(*matrix);
    *matrix = NULL;
}

void print_matrix(int64 **matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            char c = ((columns - j) > 1) ? ' ' : '\n';
            printf("%llu%c", matrix[i][j], c);
        }
    }
}

void fprint_matrix(int64 **matrix, int rows, int columns, FILE *file) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            char c = ((columns - j) > 1) ? ' ' : '\n';
            fprintf(file,"%llu%c", matrix[i][j], c);
        }
    }
}

int get_target_matrix_dimensions(int *target_row, int *target_col) {
    printf("Enter target sparse matrix rows: ");
    scanf("%d", target_row);
    if (*target_row <= 0) {
        printf("Invalid values! Number of rows must be greater than 0!\n");
        return 1;
    }

    printf("Enter target sparse matrix columns: ");
    scanf("%d", target_col);
    if (*target_col <= 0) {
        printf("Invalid values! Number of columns must be greater than 0!\n");
        return 1;
    }

    return 0;
}

int is_target_dimensions_valid(int src_row, int src_col, int target_row, int target_col) {
    if (target_row > src_row || target_col > src_col) {
        printf("Target sparse matrix can not be larger than source matrix!\n");
        return 0;
    } else if ((target_row <= 2) && (target_col <= 2)){
        printf("Any sparse matrix must have at least 3 rows or 3 columns\n");
        return 0;
    }
    return 1;
}

int64 find_sparse_matrix_seq(int64 **src_matrix, int64 **target_matrix, int64 **sum_matrix, int src_row, int src_col, int target_row, int target_col, int *target_offset_row, int *target_offset_col) {
    for (int i = 0; i <= (src_row - target_row); i++) {
        for (int j = 0; j <= (src_col - target_col); j++) {
            if (is_matrix_sparse(src_matrix, i, j, target_row, target_col)) {
                sum_matrix[i][j] = calculate_submatrix_sum_seq(src_matrix, i, j, target_row, target_col);
            } else {
                sum_matrix[i][j] = 0;
            }
        }
    }

    int max_value_row = src_row - target_row + 1;
    int max_value_col = src_col - target_col + 1;
    int max_value = matrix_max_value_pos(sum_matrix, &max_value_row, &max_value_col);

    if (max_value) {
        *target_offset_row = max_value_row;
        *target_offset_col = max_value_col;
        matrix_copy_seq(src_matrix, target_matrix, max_value_row, max_value_col, target_row, target_col);
    }

    return max_value;
}

int64 find_sparse_matrix_par(int64 **src_matrix, int64 **target_matrix, int64 **sum_matrix, int src_row, int src_col, int target_row, int target_col, int *target_offset_row, int *target_offset_col) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= (src_row - target_row); i++) {
        for (int j = 0; j <= (src_col - target_col); j++) {
            if (is_matrix_sparse(src_matrix, i, j, target_row, target_col)) {
                sum_matrix[i][j] = calculate_submatrix_sum_par(src_matrix, i, j, target_row, target_col);
            } else {
                sum_matrix[i][j] = 0;
            }
        }
    }

    int max_value_row = src_row - target_row + 1;
    int max_value_col = src_col - target_col + 1;
    int max_value = matrix_max_value_pos(sum_matrix, &max_value_row, &max_value_col);

    if (max_value) {
        *target_offset_row = max_value_row;
        *target_offset_col = max_value_col;
        matrix_copy_par(src_matrix, target_matrix, max_value_row, max_value_col, target_row, target_col);
    }

    return max_value;
}

int is_matrix_sparse(int64 **matrix, int offset_row, int offset_col, int target_row, int target_col) {
    int empty_rows = 0;
    int empty_columns = 0;
    int64 row_sum = 0;
    int64 col_sum = 0;

    for (int i = (offset_row + 1); i < (offset_row + target_row - 1); i++) {
        row_sum = 0;
        for (int j = offset_col; j < offset_col + target_col; j++) {
            row_sum += matrix[i][j];
        }
        if (row_sum == 0) {
            empty_rows += 1;
            break;
        }
    }

    if (empty_rows) {
        return 1;
    }

    for (int j = (offset_col + 1); j < (offset_col + target_col - 1); j++) {
        col_sum = 0;
        for (int i = offset_row; i < offset_row + target_row; i++) {
            col_sum += matrix[i][j];
        }
        if (col_sum == 0) {
            empty_columns += 1;
            break;
        }
    }

    if (empty_columns) {
        return 1;
    }
    return 0;
}

int64 calculate_submatrix_sum_seq(int64 **matrix, int offset_row, int offset_col, int target_row, int target_col) {
    int64 result = 0;

    for (int i = offset_row; i < offset_row + target_row; i++) {
        for (int j = offset_col; j < offset_col + target_col; j++) {
            result += matrix[i][j];
        }
    }

    return result;
}

int64 calculate_submatrix_sum_par(int64 **matrix, int offset_row, int offset_col, int target_row, int target_col) {
    int64 result = 0;

    for (int i = offset_row; i < offset_row + target_row; i++) {
        for (int j = offset_col; j < offset_col + target_col; j++) {
            #pragma omp atomic
            result += matrix[i][j];
        }
    }

    return result;
}

int64 matrix_max_value_pos(int64 **matrix, int *row, int *column) {
    int64 max_value = 0;
    int row_max = *row;
    int col_max = *column;

    for (int i = 0; i < row_max; i++) {
        for (int j = 0; j < col_max; j++) {
            if (matrix[i][j] > max_value) {
                max_value = matrix[i][j];
                *row = i;
                *column = j;
            }
        }
    }

    return max_value;
}

void matrix_copy_seq(int64 **src_matrix, int64 **target_matrix, int offset_row, int offset_col, int target_row, int target_col) {
    for (int i = 0; i < target_row; i++) {
        for (int j = 0; j < target_col; j++) {
            target_matrix[i][j] = src_matrix[offset_row + i][offset_col + j];
        }
    }
}

void matrix_copy_par(int64 **src_matrix, int64 **target_matrix, int offset_row, int offset_col, int target_row, int target_col) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < target_row; i++) {
        for (int j = 0; j < target_col; j++) {
            target_matrix[i][j] = src_matrix[offset_row + i][offset_col + j];
        }
    }
}

void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta) {
    delta->tv_nsec = finish.tv_nsec - start.tv_nsec;
    delta->tv_sec  = finish.tv_sec - start.tv_sec;

    if (delta->tv_sec > 0 && delta->tv_nsec < 0) {
        delta->tv_nsec += 1000000000;
        delta->tv_sec--;
    } else if (delta->tv_sec < 0 && delta->tv_nsec > 0) {
        delta->tv_nsec -= 1000000000;
        delta->tv_sec++;
    }
}

void print_timespec(struct timespec timestamp) {
    printf("%d.%.9ld s\n", (int) timestamp.tv_sec, timestamp.tv_nsec);
}

void fprint_timespec(struct timespec timestamp, FILE *file) {
    fprintf(file, "%d.%.9ld s\n", (int) timestamp.tv_sec, timestamp.tv_nsec);
}

void output_final_data(int64 **sum_matrix1, int64 **target_matrix1, int src_row, int src_col, int target_row,
                       int target_col, int offset_row1, int offset_col1, int64 matrix_sum1, struct timespec delta1, int64 **sum_matrix2,
                       int64 **target_matrix2, int offset_row2, int offset_col2, int64 matrix_sum2, struct timespec delta2) {
    printf("\nCalculation time (sequential): ");
    print_timespec(delta1);
    printf("\nCalculation time (parallel): ");
    print_timespec(delta2);

    printf("\nTarget matrix offset (sequential):\n");
    printf("row offset = %d, column offset = %d\n", offset_row1 + 1, offset_col1 + 1);
    printf("\nTarget matrix offset (parallel):\n");
    printf("row offset = %d, column offset = %d\n", offset_row2 + 1, offset_col2 + 1);

    printf("\nMatrix sum (sequential):\n");
    printf("%llu\n", matrix_sum1);
    printf("\nMatrix sum (parallel):\n");
    printf("%llu\n", matrix_sum2);

    printf("\nTarget matrix (sequential):\n");
    print_matrix(target_matrix1, target_row, target_col);
    printf("\nTarget matrix (parallel):\n");
    print_matrix(target_matrix2, target_row, target_col);

    printf("\nMatrix of submatrices sums (sequential):\n");
    print_matrix(sum_matrix1, src_row - target_row + 1, src_col - target_col + 1);
    printf("\nMatrix of submatrices sums (parallel):\n");
    print_matrix(sum_matrix2, src_row - target_row + 1, src_col - target_col + 1);
}

void output_final_data_file(int64 **sum_matrix1, int64 **target_matrix1, int src_row, int src_col, int target_row,
                            int target_col, int offset_row1, int offset_col1, int64 matrix_sum1, struct timespec delta1, int64 **sum_matrix2,
                            int64 **target_matrix2, int offset_row2, int offset_col2, int64 matrix_sum2, struct timespec delta2, FILE *file) {
    fprintf(file, "\nCalculation time (sequential): ");
    printf("\nCalculation time (sequential): ");
    fprint_timespec(delta1, file);
    print_timespec(delta1);
    fprintf(file, "\nCalculation time (parallel): ");
    printf("\nCalculation time (parallel): ");
    fprint_timespec(delta2, file);
    print_timespec(delta2);

    fprintf(file, "\nTarget matrix offset (sequential):\n");
    fprintf(file, "row offset = %d, column offset = %d\n", offset_row1 + 1, offset_col1 + 1);
    fprintf(file, "\nTarget matrix offset (parallel):\n");
    fprintf(file, "row offset = %d, column offset = %d\n", offset_row2 + 1, offset_col2 + 1);

    fprintf(file, "\nMatrix sum (sequential):\n");
    fprintf(file, "%llu\n", matrix_sum1);
    fprintf(file, "\nMatrix sum (parallel):\n");
    fprintf(file, "%llu\n", matrix_sum2);

    fprintf(file, "\nTarget matrix (sequential):\n");
    fprint_matrix(target_matrix1, target_row, target_col, file);
    fprintf(file, "\nTarget matrix (parallel):\n");
    fprint_matrix(target_matrix2, target_row, target_col, file);

    fprintf(file, "\nMatrix of submatrices sums (sequential):\n");
    fprint_matrix(sum_matrix1, src_row - target_row + 1, src_col - target_col + 1, file);
    fprintf(file, "\nMatrix of submatrices sums (parallel):\n");
    fprint_matrix(sum_matrix2, src_row - target_row + 1, src_col - target_col + 1, file);
}