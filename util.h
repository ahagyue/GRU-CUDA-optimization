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

double get_time();
void *read_binary(const char *filename, size_t *size);