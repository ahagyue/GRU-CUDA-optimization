#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

#include <cuda_runtime.h>

// You can modify the data structure as you want
struct Tensor {

  /* Alloc memory */
  Tensor(std::vector<int> shape_, bool isCuda_=true) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }
    
    size_t n = num_elem();
    isCuda = isCuda_;
    CHECK_CUDA(cudaMalloc(&cuda_buf, n * sizeof(float)));
    buf = (float *)malloc(n * sizeof(float));
  }

  void to_cuda(){
    size_t n = num_elem();
    CHECK_CUDA(cudaMemcpy(cuda_buf, buf, n * sizeof(float), cudaMemcpyHostToDevice));
  }

  void to_cpu(){
    size_t n = num_elem();
    CHECK_CUDA(cudaMemcpy(buf, cuda_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> shape_, float *buf_, bool isCuda_=true) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    isCuda = isCuda_;
    CHECK_CUDA(cudaMalloc(&cuda_buf, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(cuda_buf, buf_, n * sizeof(float), cudaMemcpyHostToDevice));
    buf = (float *)malloc(n * sizeof(float));
    memcpy(buf, buf_, n * sizeof(float));
  }

  ~Tensor() {
    if (buf != nullptr) {
      if (isCuda)
        CHECK_CUDA(cudaFree(cuda_buf));
      else
        free(buf);
    }
  }

  void set_zero() {
    size_t n = num_elem();
    if (isCuda)
      CHECK_CUDA(cudaMemset(cuda_buf, 0, n * sizeof(float)));
    else {
      for (size_t i = 0; i < n; i++)
        buf[i] = 0.0;
    }
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz;
  }

  // Pointer to data
  float *buf = nullptr;
  float *cuda_buf = nullptr;
  bool isCuda = true;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  size_t ndim = 0;
  size_t shape[4];
};

/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

/* Operations */

/*
 * Embedding
 * input: [BATCH] (vector)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [BATCH x EMBEDDING_DIM]
 */
__global__ void embedding1(const float *input, const float *weight, float *output) {
  int row         = threadIdx.y;
  int w_row       = input[row];
  int chunk_size  = EMBEDDING_DIM / blockDim.x;
  int start       = chunk_size * threadIdx.x;
  int end         = chunk_size * (threadIdx.x + 1);

  for(int i = start; i < end; i++) {
    output[row * EMBEDDING_DIM + i] = weight[w_row * EMBEDDING_DIM + i];
  }
}
// pthread optimization (multicore)
// void embedding2(Tensor *input, Tensor *weight, Tensor *output) {
//   size_t b = input->shape[0];
//   for(size_t i = 0; i < b; i++) {
//     CHECK_CUDA(cudaMemcpy(output->buf + (i * EMBEDDING_DIM),weight->buf + (input->buf[i] * EMBEDDING_DIM),EMBEDDING_DIM * sizeof(float),cudaMemcpyDeviceToDevice));
//   }
// }

/*
 * Reset Gate Kernel
 */
__global__ void reset_gate_kernel(const float *x, const float *h, const float *wx, const float *wh, const float *bx, const float *bh, float *output, int width_x, int width_h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int sum = 0;
  if (width_x == width_h) {
    int tiling = (width_x) / blockDim.z;
    int tiling_start = tiling * blockIdx.z;

    for(int i = tiling_start; i < tiling_start + tiling; i++) {
      sum += (x[row * width_x + i] * wx[col * width_x + i] + h[row * width_x + i] * wh[col * width_x + i]);
    }
  } else {
    int tiling_x = width_x / blockDim.z;
    int tiling_h = width_h / blockDim.z;
    int tiling_start_x = tiling_x * blockIdx.z;
    int tiling_start_h = tiling_h * blockIdx.z;
  
    for(int i = tiling_start_x; i < tiling_start_x + tiling_x; i++ ) {
      sum += x[row * width_x + i] * wx[col * width_x + i];
    }
    for (int i = tiling_start_h; i < tiling_start_h + tiling_h; i++) {
      sum += h[row * width_h + i] * wh[col * width_h + i];
    }
  }
  output[row * HIDDEN_DIM + col] = 1.0f / (1.0f + expf(- (sum + bx[col] + bh[col])));
}

/*
 * Candidate Gate Kernel
 */
__global__ void candidate_gate_kernel(const float *x, const float *h, const float *r, const float *wx, const float *wrh, const float *bx, const float *bh, float *output, int width_x, int width_h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int sum = 0;
  if (width_x == width_h) {
    int tiling = (width_x) / blockDim.z;
    int tiling_start = tiling * blockIdx.z;

    for(int i = tiling_start; i < tiling_start + tiling; i++) {
      sum += (x[row * width_x + i] * wx[col * width_x + i] + r[row * width_x + i] * h[row * width_x + i] * wrh[col * width_x + i]);
    }
  } else {
    int tiling_x = width_x / blockDim.z;
    int tiling_h = width_h / blockDim.z;
    int tiling_start_x = tiling_x * blockIdx.z;
    int tiling_start_h = tiling_h * blockIdx.z;
  
    for(int i = tiling_start_x; i < tiling_start_x + tiling_x; i++ ) {
      sum += x[row * width_x + i] * wx[col * width_x + i];
    }
    for (int i = tiling_start_h; i < tiling_start_h + tiling_h; i++) {
      sum += r[row * width_h + i] * h[row * width_h + i] * wrh[col * width_h + i];
    }
  }
  output[row * HIDDEN_DIM + col] = tanhf(sum + bx[col] + bh[col]);
}

/*
 * Final Gate Kernel
 */
 __global__ void final_gate_kernel(const float *x, const float *h, const float *wx, const float *wh, const float *bx, const float *bh, const float *g, float *output, int width_x, int width_h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int sum = 0;
  if (width_x == width_h) {
    int tiling = (width_x) / blockDim.z;
    int tiling_start = tiling * blockIdx.z;

    for(int i = tiling_start; i < tiling_start + tiling; i++) {
      sum += (x[row * width_x + i] * wx[col * width_x + i] + h[row * width_x + i] * wh[col * width_x + i]);
    }
  } else {
    int tiling_x = width_x / blockDim.z;
    int tiling_h = width_h / blockDim.z;
    int tiling_start_x = tiling_x * blockIdx.z;
    int tiling_start_h = tiling_h * blockIdx.z;
  
    for(int i = tiling_start_x; i < tiling_start_x + tiling_x; i++ ) {
      sum += x[row * width_x + i] * wx[col * width_x + i];
    }
    for (int i = tiling_start_h; i < tiling_start_h + tiling_h; i++) {
      sum += h[row * width_h + i] * wh[col * width_h + i];
    }
  }
  int pos = row * HIDDEN_DIM + col;
  output[pos] = g[pos] + (h[pos] - g[pos]) * (1.0 / (1 + expf(-(sum + bx[col] + bh[col]))));
}

/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 - x;
  }
}

/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
  }
}

/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
  }
}

/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  for (size_t i = 0; i < N_; i++) {
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1->buf[i * K_ + j] * input2->buf[j];
    }
    output->buf[i] = c;
  }
}

/*
 * SGEMM
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */
void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];
  for (size_t i = 0; i < M_; i++) {
    for (size_t j = 0; j < N_; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K_; k++) {
        c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
      }
      output->buf[i * N_ + j] = c;
    }
  }
}

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
void softmax(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  float sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    sum += expf(x);
  }
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = expf(x) / sum;
  }
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, char *parameter_fname) {

  /* Only the root process reads the parameter */
 
  size_t parameter_binary_size = 0;
  float *parameter =
      (float *)read_binary(parameter_fname, &parameter_binary_size);

  /* Network parameters */
  character_embedding =
      new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

  W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
  W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
  W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
  W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
  W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
  W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

  W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
  W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
  W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
  W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
  W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
  W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

  b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);
  b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);
  b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
  b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);
  b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);
  b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);

  b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);
  b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);
  b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
  b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);
  b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);
  b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);

  W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
  b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);

  /* input, activations, output, etc. */
  input = new Tensor({BATCH_SIZE});
  emb_out = new Tensor({BATCH_SIZE, EMBEDDING_DIM});

  hidden0 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  hidden1 = new Tensor({BATCH_SIZE, HIDDEN_DIM});

  r0 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  r1 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  z0 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  z1 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  n0 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  n1 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  f = new Tensor({BATCH_SIZE, NUM_CHAR});

  rtmp00 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  rtmp10 = new Tensor({BATCH_SIZE, HIDDEN_DIM});

  ztmp00 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp01 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp02 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp03 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp04 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp10 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp11 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp12 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp13 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ztmp14 = new Tensor({BATCH_SIZE, HIDDEN_DIM});

  ntmp00 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp01 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp02 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp03 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp04 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp05 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp10 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp11 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp12 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp13 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp14 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  ntmp15 = new Tensor({BATCH_SIZE, HIDDEN_DIM});

  htmp00 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  htmp01 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  htmp02 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  htmp10 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  htmp11 = new Tensor({BATCH_SIZE, HIDDEN_DIM});
  htmp12 = new Tensor({BATCH_SIZE, HIDDEN_DIM});

  rfloats = new Tensor({N * MAX_LEN});
  ftmp0 = new Tensor({BATCH_SIZE, NUM_CHAR});
  char_prob = new Tensor({BATCH_SIZE, NUM_CHAR});
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {
  CHECK_CUDA(cudaMemcpy(rfloats->cuda_buf, random_floats, N * MAX_LEN * sizeof(float), cudaMemcpyHostToDevice));
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));

  /* Generate N names */
  int batched_N = (N + BATCH_SIZE - 1) / BATCH_SIZE;
  for (int n = 0; n < batched_N; n++) {

    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    CHECK_CUDA(cudaMemset(input->cuda_buf, SOS, min(BATCH_SIZE, N - n * BATCH_SIZE)));
    CHECK_CUDA(cudaDeviceSynchronize());
    hidden0->set_zero();
    hidden1->set_zero();
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int l = 0; l < MAX_LEN; l++) {
      /* Embedding */
      dim3 blockDim(EMBEDDING_DIM / EMBEDDING_CHUNK, BATCH_SIZE);
      embedding1<<<1, blockDim>>>(input->cuda_buf, character_embedding->cuda_buf, emb_out->cuda_buf);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      /* First layer r */
      dim3 gridDim(HIDDEN_DIM / R_HIDDEN_PAR, BATCH_SIZE / R_BATCH_PAR);
      blockDim = dim3(R_HIDDEN_PAR, R_BATCH_PAR, R_TILING);
      reset_gate_kernel<<<gridDim, blockDim>>>(emb_out->cuda_buf, hidden0->cuda_buf, W_ir0->cuda_buf, W_hr0->cuda_buf, b_ir0->cuda_buf, b_hr0->cuda_buf, r0->cuda_buf, emb_out->shape[1], hidden0->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      /* First layer n */
      gridDim = dim3(HIDDEN_DIM / R_HIDDEN_PAR, BATCH_SIZE / R_BATCH_PAR);
      blockDim = dim3(R_HIDDEN_PAR, R_BATCH_PAR, R_TILING);
      candidate_gate_kernel<<<gridDim, blockDim>>>(emb_out->cuda_buf, hidden0->cuda_buf, r0->cuda_buf, W_in0->cuda_buf, W_hn0->cuda_buf, b_in0->cuda_buf, b_hn0->cuda_buf, n0->cuda_buf, emb_out->shape[1], hidden0->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      /* First layer h (hidden) */
      gridDim = dim3(HIDDEN_DIM / R_HIDDEN_PAR, BATCH_SIZE / R_BATCH_PAR);
      blockDim = dim3(R_HIDDEN_PAR, R_BATCH_PAR, R_TILING);
      final_gate_kernel<<<gridDim, blockDim>>>(emb_out->cuda_buf, hidden0->cuda_buf, W_iz0->cuda_buf, W_hz0->cuda_buf, b_iz0->cuda_buf, b_hz0->cuda_buf, n0->cuda_buf, z0->cuda_buf, emb_out->shape[1], hidden0->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaMemcpy(hidden0->cuda_buf, z0->cuda_buf, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

      CHECK_CUDA(cudaDeviceSynchronize());
      /* Second layer r */
      gridDim = dim3(HIDDEN_DIM / R_HIDDEN_PAR, BATCH_SIZE / R_BATCH_PAR);
      blockDim = dim3(R_HIDDEN_PAR, R_BATCH_PAR, R_TILING);
      reset_gate_kernel<<<gridDim, blockDim>>>(hidden0->cuda_buf, hidden1->cuda_buf, W_ir1->cuda_buf, W_hr1->cuda_buf, b_ir1->cuda_buf, b_hr1->cuda_buf, rtmp10->cuda_buf, hidden0->shape[1], hidden1->shape[1]);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaDeviceSynchronize());
      /* Second layer n */
      gridDim = dim3(HIDDEN_DIM / R_HIDDEN_PAR, BATCH_SIZE / R_BATCH_PAR);
      blockDim = dim3(R_HIDDEN_PAR, R_BATCH_PAR, R_TILING);
      candidate_gate_kernel<<<gridDim, blockDim>>>(hidden0->cuda_buf, hidden1->cuda_buf, r1->cuda_buf, W_in1->cuda_buf, W_hn1->cuda_buf, b_in1->cuda_buf, b_hn1->cuda_buf, n1->cuda_buf, hidden0->shape[1], hidden1->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      /* Second layer h (hidden) */
      gridDim = dim3(HIDDEN_DIM / R_HIDDEN_PAR, BATCH_SIZE / R_BATCH_PAR);
      blockDim = dim3(R_HIDDEN_PAR, R_BATCH_PAR, R_TILING);
      final_gate_kernel<<<gridDim, blockDim>>>(hidden0->cuda_buf, hidden1->cuda_buf, W_iz1->cuda_buf, W_hz1->cuda_buf, b_iz1->cuda_buf, b_hz1->cuda_buf, n1->cuda_buf, z1->cuda_buf, hidden0->shape[1], hidden1->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaMemcpy(hidden1->cuda_buf, z1->cuda_buf, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
      CHECK_CUDA(cudaDeviceSynchronize());

      hidden1->to_cpu();
      for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
          printf("%f ", hidden1->buf[i * EMBEDDING_DIM + j]);
        }
        puts("");
      }
      break;
    }
    break;
  }
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {

  delete character_embedding;
  delete W_ir0;
  delete W_iz0;
  delete W_in0;
  delete W_ir1;
  delete W_iz1;
  delete W_in1;
  delete W_hr0;
  delete W_hz0;
  delete W_hn0;
  delete W_hr1;
  delete W_hz1;
  delete W_hn1;
  delete b_ir0;
  delete b_iz0;
  delete b_in0;
  delete b_ir1;
  delete b_iz1;
  delete b_in1;
  delete b_hr0;
  delete b_hz0;
  delete b_hn0;
  delete b_hr1;
  delete b_hz1;
  delete b_hn1;
  delete W_fc;
  delete b_fc;
  delete rfloats;

  delete input;
  delete emb_out;
  delete hidden0;
  delete hidden1;
  delete r0;
  delete r1;
  delete z0;
  delete z1;
  delete n0;
  delete n1;
  delete f;
  delete char_prob;
  delete rtmp00;
  delete rtmp10;
  delete ztmp00;
  delete ztmp01;
  delete ztmp02;
  delete ztmp03;
  delete ztmp04;
  delete ztmp10;
  delete ztmp11;
  delete ztmp12;
  delete ztmp13;
  delete ztmp14;
  delete ntmp00;
  delete ntmp01;
  delete ntmp02;
  delete ntmp03;
  delete ntmp04;
  delete ntmp05;
  delete ntmp10;
  delete ntmp11;
  delete ntmp12;
  delete ntmp13;
  delete ntmp14;
  delete ntmp15;
  delete htmp00;
  delete htmp01;
  delete htmp02;
  delete htmp10;
  delete htmp11;
  delete htmp12;
  delete ftmp0;
}