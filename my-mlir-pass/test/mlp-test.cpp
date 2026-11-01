#include <cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

struct MemRef2D {
  float *allocated;
  float *data;
  std::int64_t offset;
  std::int64_t sizes[2];
  std::int64_t strides[2];
};

struct MemRef1D {
  float *allocated;
  float *data;
  std::int64_t offset;
  std::int64_t sizes[1];
  std::int64_t strides[1];
};

extern "C" void _mlir_ciface_mlp(
    const MemRef2D *x, const MemRef2D *w1, const MemRef1D *b1,
    const MemRef2D *w2, const MemRef1D *b2, MemRef2D *y);

static void checkCuda(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS)
    return;

  const char *message = nullptr;
  cuGetErrorString(status, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (message ? message : "unknown CUDA error"));
}

class PrimaryContext {
public:
  PrimaryContext() {
    checkCuda(cuInit(0), "cuInit");
    checkCuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    checkCuda(cuDevicePrimaryCtxRetain(&context, device),
              "cuDevicePrimaryCtxRetain");

    CUresult status = cuCtxPushCurrent(context);
    if (status != CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(device);
      checkCuda(status, "cuCtxPushCurrent");
    }
  }

  ~PrimaryContext() {
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
    cuDevicePrimaryCtxRelease(device);
  }

  PrimaryContext(const PrimaryContext &) = delete;
  PrimaryContext &operator=(const PrimaryContext &) = delete;

private:
  CUdevice device{};
  CUcontext context{};
};

struct CudaFree {
  void operator()(float *pointer) const {
    if (pointer)
      cuMemFree(reinterpret_cast<CUdeviceptr>(pointer));
  }
};

using Buffer = std::unique_ptr<float, CudaFree>;

static Buffer allocate(int elements) {
  CUdeviceptr pointer = 0;
  checkCuda(cuMemAllocManaged(&pointer, elements * sizeof(float),
                             CU_MEM_ATTACH_GLOBAL),
            "cuMemAllocManaged");
  return Buffer(reinterpret_cast<float *>(pointer));
}

static MemRef2D matrix(float *p) {
  return {p, p, 0, {64, 64}, {64, 1}};
}

static MemRef1D vector(float *p) {
  return {p, p, 0, {64}, {1}};
}

int main() {
  try {
    PrimaryContext context;
    auto x = allocate(4096);
    auto w1 = allocate(4096);
    auto b1 = allocate(64);
    auto w2 = allocate(4096);
    auto b2 = allocate(64);
    auto y = allocate(4096);

    for (int i = 0; i < 64; ++i) {
      b1.get()[i] = float(i % 9 - 4) / 16.0f;
      b2.get()[i] = float(i % 7 - 3) / 16.0f;
      for (int j = 0; j < 64; ++j) {
        int p = i * 64 + j;
        x.get()[p] = float((i * 7 + j * 3) % 17 - 8) / 16.0f;
        w1.get()[p] = float((i * 5 + j * 11) % 19 - 9) / 32.0f;
        w2.get()[p] = float((i * 3 + j * 7) % 13 - 6) / 32.0f;
        y.get()[p] = std::numeric_limits<float>::quiet_NaN();
      }
    }

    // CPU reference: H = ReLU(X * W1 + B1), Y = H * W2 + B2.
    std::array<double, 4096> h{};
    std::array<double, 4096> expected{};

    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 64; ++j) {
        double sum = 0.0;
        for (int k = 0; k < 64; ++k)
          sum += double(x.get()[i * 64 + k]) * double(w1.get()[k * 64 + j]);
        h[i * 64 + j] = std::max(0.0, sum + double(b1.get()[j]));
      }
    }

    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 64; ++j) {
        double sum = 0.0;
        for (int k = 0; k < 64; ++k)
          sum += h[i * 64 + k] * double(w2.get()[k * 64 + j]);
        expected[i * 64 + j] = sum + double(b2.get()[j]);
      }
    }

    auto xd = matrix(x.get());
    auto w1d = matrix(w1.get());
    auto b1d = vector(b1.get());
    auto w2d = matrix(w2.get());
    auto b2d = vector(b2.get());
    auto yd = matrix(y.get());

    auto run = [&] { _mlir_ciface_mlp(&xd, &w1d, &b1d, &w2d, &b2d, &yd); };
    auto verify = [&] {
      int mismatches = 0;
      for (int p = 0; p < 4096; ++p) {
        double actual = y.get()[p];
        double tolerance = 1e-4 * (1.0 + std::fabs(expected[p]));
        if (!std::isfinite(actual) || std::fabs(actual - expected[p]) > tolerance) {
          if (mismatches < 5)
            std::fprintf(stderr, "Mismatch [%d,%d]: expected %.9g, got %.9g\n",
                         p / 64, p % 64, expected[p], actual);
          ++mismatches;
        }
      }
      if (mismatches)
        throw std::runtime_error(std::to_string(mismatches) + " MLP mismatches");
    };

    run();
    checkCuda(cuCtxSynchronize(), "synchronize first MLP");
    verify();

    for (int i = 0; i < 10; ++i)
      run();
    checkCuda(cuCtxSynchronize(), "synchronize warmup");

    constexpr int iterations = 100;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      run();
    checkCuda(cuCtxSynchronize(), "synchronize timed MLPs");
    auto finish = std::chrono::steady_clock::now();

    verify();
    double us = std::chrono::duration<double, std::micro>(finish - start).count()
                / iterations;
    std::puts("PASS: all 4096 GPU results match the MLP reference.");
    std::printf("Mean host end-to-end MLP latency: %.3f us (%d calls).\n",
                us, iterations);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
