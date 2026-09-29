// NV5 CUDA candidate: fused residual-add + RMSNorm, N-C contract (nano-vLLM compiled baseline semantics).
//
//   s            = fp32(x) + fp32(residual)
//   residual_out = round_T(s)
//   inv          = rsqrt(sum(s * s) / H + eps)      statistic AND normalization use the UNROUNDED s
//   y            = round_T((s * inv) * fp32(w))     no round_T before the weight multiply
//
// Re-written from the CUDA Kernel Lab C6 structure (one block per row, warp-shuffle block reduction,
// 16-byte vector access). Unlike C6 (rounded-residual contract) phase 2 never reads the rounded
// residual_out: the vector path keeps s in registers; the scalar path recomputes s from x and residual,
// which is bitwise identical to phase 1. One kernel launch per call.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

constexpr unsigned FULL_MASK = 0xffffffffu;

template <int BLOCK>
__device__ __forceinline__ float block_sum(float v) {
  constexpr int NWARPS = BLOCK / 32;
  __shared__ float s_warp[NWARPS];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL_MASK, v, o);
  if (lane == 0) s_warp[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float t = lane < NWARPS ? s_warp[lane] : 0.0f;
#pragma unroll
    for (int o = NWARPS / 2; o > 0; o >>= 1) t += __shfl_down_sync(FULL_MASK, t, o);
    if (lane == 0) s_warp[0] = t;
  }
  __syncthreads();
  return s_warp[0];
}

template <typename T>
union Vec16 {
  uint4 u;
  T e[8];
};

// Vector path: H % 8 == 0, 16-byte aligned rows, H / 8 <= BLOCK * ITERS. s cached in registers.
template <typename T, int BLOCK, int ITERS>
__global__ void __launch_bounds__(BLOCK) nv5_fused_add_rmsnorm_nc_vec(
    const T* __restrict__ x, const T* __restrict__ r, const T* __restrict__ w, T* __restrict__ y,
    T* __restrict__ ro, int H, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * H;
  const int kv = H / 8;
  const uint4* xv = reinterpret_cast<const uint4*>(x + base);
  const uint4* rv = reinterpret_cast<const uint4*>(r + base);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* yv = reinterpret_cast<uint4*>(y + base);
  uint4* rov = reinterpret_cast<uint4*>(ro + base);
  float s[ITERS][8];
  float acc = 0.0f;
#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int i = threadIdx.x + it * BLOCK;
    if (i < kv) {
      Vec16<T> a, b, o;
      a.u = xv[i];
      b.u = rv[i];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s[it][j] = to_f32(a.e[j]) + to_f32(b.e[j]);
        o.e[j] = from_f32<T>(s[it][j]);
        acc += s[it][j] * s[it][j];
      }
      rov[i] = o.u;
    }
  }
  const float inv = rsqrtf(block_sum<BLOCK>(acc) / static_cast<float>(H) + eps);
#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int i = threadIdx.x + it * BLOCK;
    if (i < kv) {
      Vec16<T> g, o;
      g.u = wv[i];
#pragma unroll
      for (int j = 0; j < 8; ++j) o.e[j] = from_f32<T>((s[it][j] * inv) * to_f32(g.e[j]));
      yv[i] = o.u;
    }
  }
}

// Scalar path: any H, any alignment. Phase 2 recomputes s = fp32(x) + fp32(r) (bitwise identical).
template <typename T, int BLOCK>
__global__ void __launch_bounds__(BLOCK) nv5_fused_add_rmsnorm_nc_scalar(
    const T* __restrict__ x, const T* __restrict__ r, const T* __restrict__ w, T* __restrict__ y,
    T* __restrict__ ro, int H, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * H;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < H; i += BLOCK) {
    const float s = to_f32(x[base + i]) + to_f32(r[base + i]);
    ro[base + i] = from_f32<T>(s);
    acc += s * s;
  }
  const float inv = rsqrtf(block_sum<BLOCK>(acc) / static_cast<float>(H) + eps);
  for (int i = threadIdx.x; i < H; i += BLOCK) {
    const float s = to_f32(x[base + i]) + to_f32(r[base + i]);
    y[base + i] = from_f32<T>((s * inv) * to_f32(w[i]));
  }
}

template <typename T, int BLOCK, int ITERS>
void launch_vec(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w, at::Tensor& y, at::Tensor& ro,
                int64_t n, int h, float eps, cudaStream_t st) {
  nv5_fused_add_rmsnorm_nc_vec<T, BLOCK, ITERS><<<n, BLOCK, 0, st>>>(
      reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(r.data_ptr()),
      reinterpret_cast<const T*>(w.data_ptr()), reinterpret_cast<T*>(y.data_ptr()),
      reinterpret_cast<T*>(ro.data_ptr()), h, eps);
}

template <typename T>
void dispatch(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w, at::Tensor& y, at::Tensor& ro,
              int64_t n, int h, float eps, cudaStream_t st) {
  auto aligned = [](const at::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
  const bool vec_ok = h % 8 == 0 && aligned(x) && aligned(r) && aligned(w) && aligned(y) && aligned(ro);
  const int kv = h / 8;
  if (vec_ok && kv <= 128) {
    launch_vec<T, 128, 1>(x, r, w, y, ro, n, h, eps, st);
  } else if (vec_ok && kv <= 256) {
    launch_vec<T, 256, 1>(x, r, w, y, ro, n, h, eps, st);
  } else if (vec_ok && kv <= 512) {
    launch_vec<T, 256, 2>(x, r, w, y, ro, n, h, eps, st);
  } else if (vec_ok && kv <= 1024) {
    launch_vec<T, 256, 4>(x, r, w, y, ro, n, h, eps, st);
  } else {
    nv5_fused_add_rmsnorm_nc_scalar<T, 256><<<n, 256, 0, st>>>(
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(r.data_ptr()),
        reinterpret_cast<const T*>(w.data_ptr()), reinterpret_cast<T*>(y.data_ptr()),
        reinterpret_cast<T*>(ro.data_ptr()), h, eps);
  }
}

}  // namespace

std::vector<at::Tensor> fused_add_rms_norm_nc(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w,
                                              double eps) {
  TORCH_CHECK(x.is_cuda() && r.is_cuda() && w.is_cuda(), "nv5 cuda: tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "nv5 cuda: bf16/fp16 only");
  TORCH_CHECK(r.scalar_type() == x.scalar_type() && w.scalar_type() == x.scalar_type(), "nv5 cuda: dtype mismatch");
  TORCH_CHECK(x.dim() == 2 && r.sizes() == x.sizes() && w.dim() == 1 && w.size(0) == x.size(1),
              "nv5 cuda: expected x=[N,H], residual=[N,H], weight=[H]");
  TORCH_CHECK(x.is_contiguous() && r.is_contiguous() && w.is_contiguous(), "nv5 cuda: tensors must be contiguous");
  TORCH_CHECK(x.size(1) <= INT32_MAX, "nv5 cuda: H too large");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  auto ro = at::empty_like(x);
  const int64_t n = x.size(0);
  const int h = static_cast<int>(x.size(1));
  if (n == 0) return {y, ro};
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kBFloat16) {
    dispatch<__nv_bfloat16>(x, r, w, y, ro, n, h, static_cast<float>(eps), st);
  } else {
    dispatch<__half>(x, r, w, y, ro, n, h, static_cast<float>(eps), st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, ro};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_add_rms_norm_nc", &fused_add_rms_norm_nc, "NV5 N-C fused residual-add + RMSNorm -> (y, residual_out)");
}
