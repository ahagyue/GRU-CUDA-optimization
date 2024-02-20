#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "namegen.h"
#include "util.h"

/* Global arguments */
int rng_seed = 4155;
int N = 1;

static char *parameter_fname;
static char *output_fname;

const int print_max = 8;

void print_usage_exit(int argc, char **argv) {

  printf("Usage %s [parameter bin] [output] [N] [seed] \n", argv[0]);
  printf("  parameter bin: File conatining DNN parameters\n");
  printf("  output: File to write namegen results\n");
  printf("  N: Number of names to generate\n");
  printf("  seed: An integer RNG seed\n");

  EXIT(0);
}

void check_and_parse_args(int argc, char **argv) {
  if (argc != 5)
    print_usage_exit(argc, argv);

  int c;
  while ((c = getopt(argc, argv, "h")) != -1) {
    switch (c) {
    case 'h':
      break;
    default:
      print_usage_exit(argc, argv);
    }
  }

  parameter_fname = argv[1];
  output_fname = argv[2];
  N = atoi(argv[3]);
  rng_seed = atoi(argv[4]);
}

int main(int argc, char **argv) {

  check_and_parse_args(argc, argv);

  /* Initialize model */
  namegen_initialize(N, parameter_fname);

  float *random_floats = nullptr;
  char *output = nullptr;

  /* Initialize input and output */

  random_floats = (float *)malloc(N * MAX_LEN * sizeof(float));
  output = (char *)malloc(N * (MAX_LEN + 1) * sizeof(char));
  srand(rng_seed);
  for (int i = 0; i < N * MAX_LEN; i++) {
    random_floats[i] = ((float)rand()) / ((float)RAND_MAX);
  }

  printf("Generating %d names...", N);
  fflush(stdout);


  /* Generate names and measure time */
  double namegen_st = get_time();
  namegen(N, random_floats, output);
  double namegen_en = get_time();


  double elapsed_time = namegen_en - namegen_st;
  printf("Done!\n");

  /* Print first few result */
  int print_cnt = N < print_max ? N : print_max;
  printf("First %d results are:", print_cnt);
  for (int i = 0; i < print_cnt; i++) {
    printf(" %s%c", output + i * (MAX_LEN + 1),
            i == (print_cnt - 1) ? '\n' : ',');
  }

  /* Write the results to file */
  printf("Writing to %s ...", output_fname);
  fflush(stdout);
  FILE *output_fp = (FILE *)fopen(output_fname, "w");
  for (int i = 0; i < N; i++) {
    fprintf(output_fp, "%s\n", output + i * (MAX_LEN + 1));
  }
  fclose(output_fp);
  printf("Done!\n");

  printf("Elapsed time: %.6f seconds\n", elapsed_time);
  printf("Throughput: %.3f names/sec\n", (double)N / elapsed_time);


  /* Finalize program */
  namegen_finalize();
}