#pragma once

#include <cstdio>
#include <cstdlib>

/* Useful macros */
#define EXIT(status)                                                           \
  do {                                                                         \
    exit(status);                                                              \
  } while (0)

#define CHECK_ERROR(cond, fmt, ...)                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("[%s:%d] " fmt "\n", __FILE__, __LINE__,                          \
                       ##__VA_ARGS__);                                         \
      EXIT(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)


#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

double get_time();
void *read_binary(const char *filename, size_t *size);