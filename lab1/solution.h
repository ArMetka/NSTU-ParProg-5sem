/*
 * 64-bit unsigned integer
 * 0 - 18,446,744,073,709,551,615
 */
#define int64 unsigned long long

/*
 * Extracts filename from argv and opens file
 * 
 * return 0 in case of success
 * return 1 in case of failure
 */
int open_file(int argc, char **argv, FILE **file);

/*output.txt
 * Read matrix from file
 *
 * assumes that first line contains dimensions separated by space
 * assumes that the file is formatted correctly
 */
void get_matrix(FILE *file, int64 ***matrix, int *rows, int *columns);

/*
 * Allocate matrix of rows by columns size
 */
void allocate_matrix(int64 ***matrix, int rows, int columns);

/*
 * Free matrix of rows by columns size
 */
void free_matrix(int64 ***matrix, int rows);

/*
 * Print matrix to stdout
 */
void print_matrix(int64 **matrix, int rows, int columns);

/*
 * Print matrix to file
 */
void fprint_matrix(int64 **matrix, int rows, int columns, FILE *file);

/*
 * Get target matrix dimensions from stdin
 *
 * number of rows/columns must be greater than 0
 * 
 * return 0 in case of success
 * return 1 in case of failure
 */
int get_target_matrix_dimensions(int *target_row, int *target_col);

/*
 * Check if target matrix dimensions are valid
 *
 * dimensions are valid if target matrix smaller or equal than source matrix
 * and if target matrix can be sparse
 * matrix can be sparse if it can have at least 1 row/column
 * that consists of only zeroes and that row/column isn't first or last
 * 
 * return 1 in case of success (valid)
 * return 0 in case of failure (invalid)
 */
int is_target_dimensions_valid(int src_row, int src_col, int target_row, int target_col);

/*
 * Find sparse matrix with max sum of elements of target_row by target_col size
 *
 * check every possible target matrix position
 * check if matrix is sparse
 * write matrix sum to sum_matrix (write 0 if matrix not sparse)
 * 
 * return 0 in case of failure (max sum = 0)
 * return sum in case of success
 * 
 * sequential
 */
int64 find_sparse_matrix_seq(int64 **src_matrix, int64 **target_matrix, int64 **sum_matrix, int src_row, int src_col, int target_row, int target_col, int *target_offset_row, int *target_offset_col);

/*
 * Find sparse matrix with max sum of elements of target_row by target_col size
 *
 * check every possible target matrix position
 * check if matrix is sparse
 * write matrix sum to sum_matrix (write 0 if matrix not sparse)
 * 
 * return 0 in case of failure (max sum = 0)
 * return sum in case of success
 * 
 * parallel
 */
int64 find_sparse_matrix_par(int64 **src_matrix, int64 **target_matrix, int64 **sum_matrix, int src_row, int src_col, int target_row, int target_col, int *target_offset_row, int *target_offset_col);

/*
 * Check if matrix is sparse
 * sparse matrix -> at least one row/column must be all zeroes (but not the first/last row/column)
 */
int is_matrix_sparse(int64 **matrix, int offset_row, int offset_col, int target_row, int target_col);

/*
 * Calculate submatrix sum
 *
 * return sum
 * 
 * sequential
 */
int64 calculate_submatrix_sum_seq(int64 **matrix, int offset_row, int offset_col, int target_row, int target_col);

/*
 * Calculate submatrix sum
 *
 * return sum
 * 
 * parallel
 */
int64 calculate_submatrix_sum_par(int64 **matrix, int offset_row, int offset_col, int target_row, int target_col);

/*
 * Find row and column of matrix max value
 *
 * row/column - number of rows/columns in matrix
 * after execution
 * row/column - row and column of max value in matrix
 * 
 * return 0 in case of failure (max_value = 0)
 * return sum in case of success
 */
int64 matrix_max_value_pos(int64 **matrix, int *row, int *column);

/*
 * Copy matrix of size target_row by target_col
 * with row offset of offset_row and column offset of offset_col
 * from src_matrix to target_matrix
 * 
 * sequential
 */
void matrix_copy_seq(int64 **src_matrix, int64 **target_matrix, int offset_row, int offset_col, int target_row, int target_col);

/*
 * Copy matrix of size target_row by target_col
 * with row offset of offset_row and column offset of offset_col
 * from src_matrix to target_matrix
 * 
 * parallel
 */
void matrix_copy_par(int64 **src_matrix, int64 **target_matrix, int offset_row, int offset_col, int target_row, int target_col);

/*
 * Get difference between 2 timestamps (struct timespec)
 */
void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta);

/*
 * Print timestamp to stdout (struct timespec)
 */
void print_timespec(struct timespec timestamp);

/*
 * Print timestamp to file (struct timespec)
 */
void fprint_timespec(struct timespec timestamp, FILE *file);

/*
 * Output all info to stdout
 */
void output_final_data(int64 **sum_matrix1, int64 **target_matrix1, int src_row, int src_col, int target_row,
                       int target_col, int offset_row1, int offset_col1, int64 matrix_sum1, struct timespec delta1, int64 **sum_matrix2,
                       int64 **target_matrix2, int offset_row2, int offset_col2, int64 matrix_sum2, struct timespec delta2);

/*
 * Output all info to file
 */
void output_final_data_file(int64 **sum_matrix1, int64 **target_matrix1, int src_row, int src_col, int target_row,
                            int target_col, int offset_row1, int offset_col1, int64 matrix_sum1, struct timespec delta1, int64 **sum_matrix2,
                            int64 **target_matrix2, int offset_row2, int offset_col2, int64 matrix_sum2, struct timespec delta2, FILE *file);