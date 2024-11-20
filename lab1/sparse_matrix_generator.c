#include <stdio.h>
#include <stdlib.h>

int file_size_warning(int max_value, int rows, int columns, int zero_probability);

/*
 * ./generator [FILENAME]
 * 
 * Generates matrix of size [rows] by [columns]
 * first line contains numbers of rows and columns separated by space
 * every element have ([zero_probability] / 100) probability of becoming '0'
 * every element is less or equal than [max_value]
 * RNG is controlled by srand([seed])
 * output matrix to file if [FILENAME] specified, otherwise output matrix to stdout
 */
int main(int argc, char **argv) {
    int rows = 0;
    int columns = 0;
    int zero_probability = 0;
    int max_value = 0;
    int seed = 0;
    FILE *file = NULL;

    if (argc > 1) {
        char filename[256];
        for(int count = 0; *(*(argv + 1) + count) != '\0'; count++) {
            filename[count] = *(*(argv + 1) + count);
            filename[count + 1] = '\0';
        }
        file = fopen(filename, "w");
        if (!file) {
            printf("Failed to create file!\n");
            return 1;
        }
    }

    printf("Enter matrix dimensions in format \"rows columns\": ");
    scanf("%d %d", &rows, &columns);
    if (rows <= 0 || columns <= 0) {
        printf("Invalid values! Number of rows and columns must be greater than 0!\n");
        if (file) {
            fclose(file);
        }
        return 1;
    }

    printf("Enter \'0\' probability (integer [0 - 100]%%): ");
    scanf("%d", &zero_probability);
    if (zero_probability < 0 || zero_probability > 100) {
        printf("Invalid values! Zero value probability must be in range [0 - 100]%%!\n");
        if (file) {
            fclose(file);
        }
        return 1;
    }

    printf("Enter max value (positive integer): ");
    scanf("%d", &max_value);
    if (max_value <= 0) {
        printf("Invalid values! Max value must be greater than 0!\n");
        if (file) {
            fclose(file);
        }
        return 1;
    }

    printf("Enter seed (integer): ");
    scanf("%d", &seed);
    srand(seed);

    int size_warning = file_size_warning(max_value, rows, columns, zero_probability);

    if (size_warning) {
        if (file) {
            fclose(file);
        }
        return 1;
    }

    if (file) {
        fprintf(file, "%d %d\n", rows, columns);
    } else {
        printf("%d %d\n", rows, columns);
    }
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            int random_num = rand();
            if ((random_num % 100) < zero_probability) {
                random_num = 0;
            } else {
                while (random_num > max_value) {
                    random_num /= 10;
                }
            }

            char last_c = ((columns - j) > 1) ? ' ' : '\n';

            if (file) {
                fprintf(file, "%d%c", random_num, last_c);
            } else {
                printf("%d%c", random_num, last_c);
            }
        }
    }

    if (file) {
        printf("Done! [FILE OUTPUT]\n");
        fclose(file);
    }

    return 0;
}

/*
 * row * col * 1 -> spaces
 * row -> \n
 * row * col * (prob / 100) -> 0
 * row * col * (1 - prob / 100) * (max_value_places) -> everything else
 *
 * (row * col * 1) + (row) + (row * col * (prob / 100)) + (row * col * (1 - prob / 100) * (max_value_places))
 * 
 * return 0 in case of success (continue)
 * return 1 in case of failure (exit)
 */
int file_size_warning(int max_value, int rows, int columns, int zero_probability) {
    int max_value_places = 1;
    while ((max_value / 10) > 0) {
        max_value /= 10;
        max_value_places++;
    }
    double estimated_file_size = (rows * columns) + rows + (rows * columns * (((double) zero_probability) / 100.0)) +
                                 (rows * columns * (1 - ((double) zero_probability) / 100.0) * ((double) max_value_places));
    printf("Estimated file size: \e[31m%.0lf KB\e[0m\n", estimated_file_size / 1024.0 + 1);
    printf("Are you sure you want to continue file generation? [yes/no]: ");
    char answer[64];
    scanf("%s", answer);

    if ((answer[0] == 'y' || answer[0] == 'Y') &&
        (answer[1] == 'e' || answer[1] == 'E') &&
        (answer[2] == 's' || answer[2] == 'S')) {
            return 0;
        } else {
            return 1;
        }
}
