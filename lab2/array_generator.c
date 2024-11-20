#include <stdio.h>
#include <stdlib.h>

#define uint64 unsigned long long
#define uint32 unsigned int

int file_size_warning(uint32 array_length, uint32 max_value);

int main(int argc, char **argv) {
    uint32 array_length = 0;
    uint32 max_value = 0;
    uint32 seed = 0;
    FILE *file = NULL;

    if (argc > 1) {
        char filename[256];
        for (int count = 0; *(*(argv + 1) + count) != '\0'; count++) {
            filename[count] = *(*(argv + 1) + count);
            filename[count + 1] = '\0';
        }
        file = fopen(filename, "w");
        if (!file) {
            printf("Failed to create file!\n");
            return 1;
        }
    } else {
        printf("No file specified!\n");
        return 1;
    }

    do {
        printf("Enter array length (unsigned integer): ");
        scanf("%u", &array_length);

        printf("Enter max value (unsigned integer <= 2^31): ");
        scanf("%u", &max_value);

        printf("Enter seed (unsigned integer): ");
        scanf("%d", &seed);
        srand(seed);
    } while (file_size_warning(array_length, max_value));

    for (uint32 i = 0; i < array_length; i++) {
        uint32 random_num = rand();
        while (random_num > max_value) {
            random_num /= 10;
        }
        char last_c = (i == (array_length - 1)) ? '\n' : ' ';
        fprintf(file, "%d%c", random_num, last_c);
    }

    fclose(file);
}

/**
 * array_length -> spaces
 * array_length * max_value_places -> everything else
 *
 * array_length + (array_length * max_value_places)
 *
 * @return 0 in case of success (continue);
 * @return 1 in case of failure (exit)
 */
int file_size_warning(uint32 array_length, uint32 max_value) {
    int max_value_places = 1;
    while ((max_value / 10) > 0) {
        max_value /= 10;
        max_value_places++;
    }

    double estimated_file_size = ((double)array_length + ((double)array_length * (double)max_value_places));
    printf("Estimated file size: \e[31m%.0lf KB\e[0m\n", estimated_file_size / 1024.0 + 1);
    printf("Are you sure you want to continue file generation? [yes/no]: ");
    char answer[64];
    scanf("%s", answer);

    if ((answer[0] == 'y' || answer[0] == 'Y') && (answer[1] == 'e' || answer[1] == 'E') &&
        (answer[2] == 's' || answer[2] == 'S')) {
        return 0;
    } else {
        return 1;
    }
}
