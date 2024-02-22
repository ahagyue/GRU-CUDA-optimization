#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

nvtxRangeId_t nvtx_range_start(const char *message) {
  nvtxEventAttributes_t eventAttrib={0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  eventAttrib.color = 0xFF800080;
  return nvtxRangeStartEx(&eventAttrib);
}
void nvtx_range_stop(nvtxRangeId_t nvtx_id) {
  nvtxRangeEnd(nvtx_id);
}


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
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  output[row * EMBEDDING_DIM + col] = weight[((int)input[row]) * EMBEDDING_DIM + col];
}

/*
 * Reset Gate Kernel
 */
__global__ void reset_gate_kernel(const float *x, const float *h, const float *wx, const float *wh, const float *bx, const float *bh, float *output, int width_x, int width_h) {
  int gj = blockIdx.x, gi = blockIdx.y;
  int lj = threadIdx.x, li = threadIdx.y;

  int col = gj * blockDim.x + lj;
  int row = gi * blockDim.y + li;

  __shared__ float xlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float wxlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float hlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float whlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];

  float sum = bx[col] + bh[col];
  if (width_x == width_h) {
    for(int i = 0; i < width_x; i += BLOCK_SIZE) {
      xlocal[li][lj] = x[(row) * width_x + (i + lj)];
      wxlocal[li][lj] = wx[(gj*BLOCK_SIZE+li) * width_x + (i+lj)];
      hlocal[li][lj] = h[(row) * width_h + (i + lj)];
      whlocal[li][lj] = wh[(gj*BLOCK_SIZE+li) * width_h + (i+lj)];

      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += xlocal[li][j] * wxlocal[lj][j] + hlocal[li][j] * whlocal[lj][j];
      }
      __syncthreads();
    }
  } else {
    for(int i = 0; i < width_x; i += BLOCK_SIZE) {
      xlocal[li][lj] = x[(row) * width_x + (i + lj)];
      wxlocal[li][lj] = wx[(gj*BLOCK_SIZE+li) * width_x + (i+lj)];
      
      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += xlocal[li][j] * wxlocal[lj][j];
      }
      __syncthreads();
    }
    for (int i = 0; i < width_h; i += BLOCK_SIZE) {
      hlocal[li][lj] = h[(row) * width_h + (i + lj)];
      whlocal[li][lj] = wh[(gj*BLOCK_SIZE+li) * width_h + (i+lj)];

      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += hlocal[li][j] * whlocal[lj][j];
      }
      __syncthreads();
    }
  }

  output[row * HIDDEN_DIM + col] = 1.0 / (1.0 + expf(-sum));
}

/*
 * Candidate Gate Kernel
 */
__global__ void candidate_gate_kernel(const float *x, const float *h, const float *r, const float *wx, const float *wrh, const float *bx, const float *bh, float *output, int width_x, int width_h) {
  int gj = blockIdx.x, gi = blockIdx.y;
  int lj = threadIdx.x, li = threadIdx.y;

  int col = gj * blockDim.x + lj;
  int row = gi * blockDim.y + li;

  __shared__ float xlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float wxlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float hlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float wrhlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];

  float _r = r[row * width_h + col];

  float sum = 0;
  if (width_x == width_h) {
    for(int i = 0; i < width_x; i += BLOCK_SIZE) {
      xlocal[li][lj] = x[(row) * width_x + (i + lj)];
      wxlocal[li][lj] = wx[(gj*BLOCK_SIZE+li) * width_x + (i+lj)];
      hlocal[li][lj] = h[(row) * width_h + (i + lj)];
      wrhlocal[li][lj] = wrh[(gj*BLOCK_SIZE+li) * width_h + (i+lj)];

      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += xlocal[li][j] * wxlocal[lj][j] + _r * hlocal[li][j] * wrhlocal[lj][j];
      }
      __syncthreads();
    }
  } else {
    for(int i = 0; i < width_x; i += BLOCK_SIZE) {
      xlocal[li][lj] = x[(row) * width_x + (i + lj)];
      wxlocal[li][lj] = wx[(gj*BLOCK_SIZE+li) * width_x + (i+lj)];
      
      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += xlocal[li][j] * wxlocal[lj][j];
      }
      __syncthreads();
    }
    for (int i = 0; i < width_h; i += BLOCK_SIZE) {
      hlocal[li][lj] = h[(row) * width_h + (i + lj)];
      wrhlocal[li][lj] = wrh[(gj*BLOCK_SIZE+li) * width_h + (i+lj)];

      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += _r * hlocal[li][j] * wrhlocal[lj][j];
      }
      __syncthreads();
    }
  }
  output[row * HIDDEN_DIM + col] = tanhf(sum + bx[col] + r[row * width_h + col] * bh[col]);
}

/*
 * Final Gate Kernel
 */
 __global__ void final_gate_kernel(const float *x, const float *h, const float *wx, const float *wh, const float *bx, const float *bh, const float *g, float *output, int width_x, int width_h) {
  int gj = blockIdx.x, gi = blockIdx.y;
  int lj = threadIdx.x, li = threadIdx.y;

  int col = gj * blockDim.x + lj;
  int row = gi * blockDim.y + li;

  __shared__ float xlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float wxlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float hlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float whlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];

  float sum = bx[col] + bh[col];
  if (width_x == width_h) {
    for(int i = 0; i < width_x; i += BLOCK_SIZE) {
      xlocal[li][lj] = x[(row) * width_x + (i + lj)];
      wxlocal[li][lj] = wx[(gj*BLOCK_SIZE+li) * width_x + (i+lj)];
      hlocal[li][lj] = h[(row) * width_h + (i + lj)];
      whlocal[li][lj] = wh[(gj*BLOCK_SIZE+li) * width_h + (i+lj)];

      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += xlocal[li][j] * wxlocal[lj][j] + hlocal[li][j] * whlocal[lj][j];
      }
      __syncthreads();
    }
  } else {
    for(int i = 0; i < width_x; i += BLOCK_SIZE) {
      xlocal[li][lj] = x[(row) * width_x + (i + lj)];
      wxlocal[li][lj] = wx[(gj*BLOCK_SIZE+li) * width_x + (i+lj)];
      
      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += xlocal[li][j] * wxlocal[lj][j];
      }
      __syncthreads();
    }
    for (int i = 0; i < width_h; i += BLOCK_SIZE) {
      hlocal[li][lj] = h[(row) * width_h + (i + lj)];
      whlocal[li][lj] = wh[(gj*BLOCK_SIZE+li) * width_h + (i+lj)];

      __syncthreads();

      for(int j = 0; j < BLOCK_SIZE; j++) {
        sum += hlocal[li][j] * whlocal[lj][j];
      }
      __syncthreads();
    }
  }
  int pos = row * HIDDEN_DIM + col;
  output[pos] = g[pos] + (h[pos] - g[pos]) * (1.0 / (1.0 + expf(-sum)));
}

/*
 * Matrix Multiplication
 */
__global__ void matmul_kernel(const float *x , const float *w, const float *b, float *output, int M, int N, int K) {
  int gj = blockIdx.x, gi = blockIdx.y;
  int lj = threadIdx.x, li = threadIdx.y;

  int col = gj * blockDim.x + lj;
  int row = gi * blockDim.y + li;

  __shared__ float xlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];
  __shared__ float wlocal[BLOCK_SIZE][BLOCK_SIZE+SHM_PADDING];

  float sum = b[col];
  for(int i = 0; i < K; i+=BLOCK_SIZE) {
    xlocal[li][lj] = x[(row) * K + (i + lj)];
    wlocal[li][lj] = w[(gj*BLOCK_SIZE+li) * K + (i+lj)];
    
    __syncthreads();

    for(int j = 0; j < BLOCK_SIZE; j++) {
      sum += xlocal[li][j] * wlocal[lj][j];
    }
    __syncthreads();
  }
  output[row * N + col] = sum;
}

/*
 * Reduce Sum of Exponent
 */
__global__ void reduce_exp_sum_kernel(float *input, float *output) {

}

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
void softmax(Tensor *input, Tensor *output) {
  size_t M = input->shape[0];
  size_t N = input->shape[1];
  for(size_t y = 0; y < M; y++) {
    float sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      float x = input->buf[y*N+i];
      sum += expf(x);
    }
    for (size_t i = 0; i < N; i++) {
      float x = input->buf[y*N+i];
      output->buf[y*N+i] = expf(x) / sum;
    }
  }
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset, int in_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->shape[1];
  float psum = 0.0;
  for (size_t i = in_offset; i < in_offset + NUM_CHAR; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i - in_offset;
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
  f = new Tensor({NUM_CHAR});

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
    for(int i = 0; i < BATCH_SIZE; i++) input->buf[i] = SOS;
    input->to_cuda();
    hidden0->set_zero();
    hidden1->set_zero();

    for (int l = 0; l < MAX_LEN; l++) {
      /* Embedding */
      dim3 gridDim(EMBEDDING_DIM / EMBEDDING_PAR, BATCH_SIZE / E_BATCH_PAR);
      dim3 blockDim(EMBEDDING_PAR, E_BATCH_PAR);
      embedding1<<<gridDim, blockDim>>>(input->cuda_buf, character_embedding->cuda_buf, emb_out->cuda_buf);
      CHECK_CUDA(cudaGetLastError());

      /* First layer r */
      gridDim = dim3(HIDDEN_DIM / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      reset_gate_kernel<<<gridDim, blockDim>>>(emb_out->cuda_buf, hidden0->cuda_buf, W_ir0->cuda_buf, W_hr0->cuda_buf, b_ir0->cuda_buf, b_hr0->cuda_buf, r0->cuda_buf, emb_out->shape[1], hidden0->shape[1]);
      CHECK_CUDA(cudaGetLastError());

      /* First layer n */
      gridDim = dim3(HIDDEN_DIM / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      candidate_gate_kernel<<<gridDim, blockDim>>>(emb_out->cuda_buf, hidden0->cuda_buf, r0->cuda_buf, W_in0->cuda_buf, W_hn0->cuda_buf, b_in0->cuda_buf, b_hn0->cuda_buf, n0->cuda_buf, emb_out->shape[1], hidden0->shape[1]);
      CHECK_CUDA(cudaGetLastError());

      /* First layer h (hidden) */
      gridDim = dim3(HIDDEN_DIM / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      final_gate_kernel<<<gridDim, blockDim>>>(emb_out->cuda_buf, hidden0->cuda_buf, W_iz0->cuda_buf, W_hz0->cuda_buf, b_iz0->cuda_buf, b_hz0->cuda_buf, n0->cuda_buf, z0->cuda_buf, emb_out->shape[1], hidden0->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaMemcpy(hidden0->cuda_buf, z0->cuda_buf, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

      /* Second layer r */
      gridDim = dim3(HIDDEN_DIM / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      reset_gate_kernel<<<gridDim, blockDim>>>(hidden0->cuda_buf, hidden1->cuda_buf, W_ir1->cuda_buf, W_hr1->cuda_buf, b_ir1->cuda_buf, b_hr1->cuda_buf, r1->cuda_buf, hidden0->shape[1], hidden1->shape[1]);
      CHECK_CUDA(cudaGetLastError());

      /* Second layer n */
      gridDim = dim3(HIDDEN_DIM / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      candidate_gate_kernel<<<gridDim, blockDim>>>(hidden0->cuda_buf, hidden1->cuda_buf, r1->cuda_buf, W_in1->cuda_buf, W_hn1->cuda_buf, b_in1->cuda_buf, b_hn1->cuda_buf, n1->cuda_buf, hidden0->shape[1], hidden1->shape[1]);
      CHECK_CUDA(cudaGetLastError());

      /* Second layer h (hidden) */
      gridDim = dim3(HIDDEN_DIM / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      final_gate_kernel<<<gridDim, blockDim>>>(hidden0->cuda_buf, hidden1->cuda_buf, W_iz1->cuda_buf, W_hz1->cuda_buf, b_iz1->cuda_buf, b_hz1->cuda_buf, n1->cuda_buf, z1->cuda_buf, hidden0->shape[1], hidden1->shape[1]);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaMemcpy(hidden1->cuda_buf, z1->cuda_buf, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
      
      /* Fully connected layer */
      gridDim = dim3(NUM_CHAR / BLOCK_SIZE, BATCH_SIZE / BLOCK_SIZE);
      blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
      matmul_kernel<<<gridDim, blockDim>>>(hidden1->cuda_buf, W_fc->cuda_buf, b_fc->cuda_buf, ftmp0->cuda_buf, BATCH_SIZE, NUM_CHAR, HIDDEN_DIM);
      CHECK_CUDA(cudaGetLastError());

      ftmp0->to_cpu();
      rfloats->to_cpu();
      
      nvtxRangeId_t cpuid = nvtx_range_start("cpu range");
      softmax(ftmp0, char_prob);
      for(int i = 0; i < BATCH_SIZE; i++) {
        /* Random select */
        int selected_char = random_select(char_prob, rfloats, (n * BATCH_SIZE + i) * MAX_LEN + l, i * NUM_CHAR);
        output[(n * BATCH_SIZE + i) * (MAX_LEN + 1) + l] = selected_char;
        input->buf[i] = (float)selected_char;
      }
      input->to_cuda();
      nvtx_range_stop(cpuid);
    }
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
  delete ftmp0;
}