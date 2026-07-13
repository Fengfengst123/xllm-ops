#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <strings.h>
#include <string>
#include <vector>

#define CHECK_ACL(expr)                                                        \
  do {                                                                         \
    auto _ret = (expr);                                                        \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "%s failed, ret=%d, line=%d\n", #expr,             \
                   static_cast<int>(_ret), __LINE__);                         \
      return false;                                                            \
    }                                                                          \
  } while (0)

using GammaGetWorkspaceFn = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                            const aclTensor *, double, bool,
                                            aclTensor *, aclTensor *,
                                            aclTensor *, uint64_t *,
                                            aclOpExecutor **);
using AddGetWorkspaceFn = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                          const aclTensor *, double, aclTensor *,
                                          aclTensor *, aclTensor *, uint64_t *,
                                          aclOpExecutor **);
using RunFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AddsGetWorkspaceFn = aclnnStatus (*)(const aclTensor *, const aclScalar *,
                                           const aclScalar *, aclTensor *,
                                           uint64_t *, aclOpExecutor **);
using AddsRunFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

struct GammaApi {
  void *handle = nullptr;
  GammaGetWorkspaceFn get_ws = nullptr;
  RunFn run = nullptr;
};

struct AddApi {
  void *handle = nullptr;
  AddGetWorkspaceFn get_ws = nullptr;
  RunFn run = nullptr;
};

struct AddsApi {
  void *handle = nullptr;
  AddsGetWorkspaceFn get_ws = nullptr;
  AddsRunFn run = nullptr;
};

struct Tensor {
  void *dev = nullptr;
  aclTensor *desc = nullptr;
  std::vector<int64_t> shape;
  size_t bytes = 0;
};

struct CaseConfig {
  const char *name;
  aclDataType dtype;
  size_t elem_size;
  double atol;
};

static int64_t GetEnvI64(const char *name, int64_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  long long parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed <= 0) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

static int GetEnvInt(const char *name, int fallback) {
  return static_cast<int>(GetEnvI64(name, fallback));
}

static bool EnvEnabled(const char *name, bool fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
         strcasecmp(value, "yes") == 0;
}

static int64_t Numel(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

static std::vector<int64_t> Strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

static bool CreateTensor(const std::vector<uint8_t> &host,
                         const std::vector<int64_t> &shape, aclDataType dtype,
                         size_t elem_size, Tensor *tensor) {
  tensor->shape = shape;
  tensor->bytes = static_cast<size_t>(Numel(shape)) * elem_size;
  CHECK_ACL(aclrtMalloc(&tensor->dev, tensor->bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tensor->dev, tensor->bytes, host.data(), tensor->bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  auto strides = Strides(shape);
  tensor->desc = aclCreateTensor(shape.data(), shape.size(), dtype,
                                 strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), tensor->dev);
  if (tensor->desc == nullptr) {
    std::fprintf(stderr, "aclCreateTensor failed\n");
    return false;
  }
  return true;
}

static bool CreateZeroTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                             size_t elem_size, Tensor *tensor) {
  std::vector<uint8_t> zeros(static_cast<size_t>(Numel(shape)) * elem_size, 0);
  return CreateTensor(zeros, shape, dtype, elem_size, tensor);
}

static void DestroyTensor(Tensor *tensor) {
  if (tensor->desc != nullptr) {
    aclDestroyTensor(tensor->desc);
    tensor->desc = nullptr;
  }
  if (tensor->dev != nullptr) {
    aclrtFree(tensor->dev);
    tensor->dev = nullptr;
  }
}

static uint16_t FloatToBf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static float Bf16ToFloat(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t FloatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFFU;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  uint32_t rounded = mant + 0x1000U;
  if (rounded & 0x800000U) {
    rounded = 0;
    ++exp;
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (rounded >> 13));
}

static float HalfToFloat(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000U)) << 16;
  uint32_t exp = (h >> 10) & 0x1FU;
  uint32_t mant = h & 0x3FFU;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x400U) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFU;
      bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static void PutU16(std::vector<uint8_t> *data, size_t idx, uint16_t value) {
  (*data)[2 * idx] = static_cast<uint8_t>(value & 0xFFU);
  (*data)[2 * idx + 1] = static_cast<uint8_t>(value >> 8);
}

static uint16_t GetU16(const std::vector<uint8_t> &data, size_t idx) {
  return static_cast<uint16_t>(data[2 * idx]) |
         (static_cast<uint16_t>(data[2 * idx + 1]) << 8);
}

static void PutF32(std::vector<uint8_t> *data, size_t idx, float value) {
  std::memcpy(data->data() + idx * sizeof(float), &value, sizeof(value));
}

static float GetF32(const std::vector<uint8_t> &data, size_t idx) {
  float value;
  std::memcpy(&value, data.data() + idx * sizeof(float), sizeof(value));
  return value;
}

static std::vector<uint8_t> MakeData(const CaseConfig &cfg,
                                     const std::vector<int64_t> &shape,
                                     float value) {
  const size_t n = static_cast<size_t>(Numel(shape));
  std::vector<uint8_t> data(n * cfg.elem_size, 0);
  if (cfg.dtype == ACL_FLOAT) {
    for (size_t i = 0; i < n; ++i) {
      PutF32(&data, i, value);
    }
  } else if (cfg.dtype == ACL_FLOAT16) {
    uint16_t h = FloatToHalf(value);
    for (size_t i = 0; i < n; ++i) {
      PutU16(&data, i, h);
    }
  } else {
    uint16_t b = FloatToBf16(value);
    for (size_t i = 0; i < n; ++i) {
      PutU16(&data, i, b);
    }
  }
  return data;
}

static float PatternValue(size_t idx, float lo, float hi, uint32_t salt) {
  uint32_t x = static_cast<uint32_t>(idx) * 747796405U + 2891336453U + salt;
  x ^= x >> 16;
  x *= 2246822519U;
  x ^= x >> 13;
  const float unit = static_cast<float>(x & 0xFFFFU) / 65535.0f;
  return lo + (hi - lo) * unit;
}

static std::vector<uint8_t> MakePatternData(const CaseConfig &cfg,
                                            const std::vector<int64_t> &shape,
                                            float lo, float hi,
                                            uint32_t salt) {
  const size_t n = static_cast<size_t>(Numel(shape));
  std::vector<uint8_t> data(n * cfg.elem_size, 0);
  for (size_t i = 0; i < n; ++i) {
    float value = PatternValue(i, lo, hi, salt);
    if (cfg.dtype == ACL_FLOAT) {
      PutF32(&data, i, value);
    } else if (cfg.dtype == ACL_FLOAT16) {
      PutU16(&data, i, FloatToHalf(value));
    } else {
      PutU16(&data, i, FloatToBf16(value));
    }
  }
  return data;
}

static std::vector<uint8_t> MakeGammaPlusData(const CaseConfig &cfg,
                                              const std::vector<uint8_t> &gamma) {
  const size_t n = gamma.size() / cfg.elem_size;
  std::vector<uint8_t> data(gamma.size(), 0);
  for (size_t i = 0; i < n; ++i) {
    if (cfg.dtype == ACL_FLOAT) {
      PutF32(&data, i, GetF32(gamma, i) + 1.0f);
    } else if (cfg.dtype == ACL_FLOAT16) {
      PutU16(&data, i, FloatToHalf(HalfToFloat(GetU16(gamma, i)) + 1.0f));
    } else {
      PutU16(&data, i, FloatToBf16(Bf16ToFloat(GetU16(gamma, i)) + 1.0f));
    }
  }
  return data;
}

static double ValueAsFloat(const CaseConfig &cfg, const std::vector<uint8_t> &data,
                           size_t idx) {
  if (cfg.dtype == ACL_FLOAT) {
    return GetF32(data, idx);
  }
  if (cfg.dtype == ACL_FLOAT16) {
    return HalfToFloat(GetU16(data, idx));
  }
  return Bf16ToFloat(GetU16(data, idx));
}

static bool LoadGammaApi(const char *lib_path, GammaApi *api) {
  api->handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
  if (api->handle == nullptr) {
    std::fprintf(stderr, "dlopen %s failed: %s\n", lib_path, dlerror());
    return false;
  }
  api->get_ws = reinterpret_cast<GammaGetWorkspaceFn>(
      dlsym(api->handle, "aclnnGammaAddRmsNormGetWorkspaceSize"));
  api->run = reinterpret_cast<RunFn>(dlsym(api->handle, "aclnnGammaAddRmsNorm"));
  if (api->get_ws == nullptr || api->run == nullptr) {
    std::fprintf(stderr, "dlsym failed in %s: aclnnGammaAddRmsNorm\n",
                 lib_path);
    return false;
  }
  return true;
}

static bool LoadAddApi(const char *lib_path, AddApi *api) {
  api->handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
  if (api->handle == nullptr) {
    std::fprintf(stderr, "dlopen %s failed: %s\n", lib_path, dlerror());
    return false;
  }
  api->get_ws = reinterpret_cast<AddGetWorkspaceFn>(
      dlsym(api->handle, "aclnnAddRmsNormGetWorkspaceSize"));
  api->run = reinterpret_cast<RunFn>(dlsym(api->handle, "aclnnAddRmsNorm"));
  if (api->get_ws == nullptr || api->run == nullptr) {
    std::fprintf(stderr, "dlsym failed in %s: aclnnAddRmsNorm\n", lib_path);
    return false;
  }
  return true;
}

static bool LoadAddsApi(const char *lib_path, AddsApi *api) {
  api->handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
  if (api->handle == nullptr) {
    std::fprintf(stderr, "dlopen %s failed: %s\n", lib_path, dlerror());
    return false;
  }
  api->get_ws = reinterpret_cast<AddsGetWorkspaceFn>(
      dlsym(api->handle, "aclnnAddsGetWorkspaceSize"));
  api->run = reinterpret_cast<AddsRunFn>(dlsym(api->handle, "aclnnAdds"));
  if (api->get_ws == nullptr || api->run == nullptr) {
    std::fprintf(stderr, "dlsym failed in %s: aclnnAdds\n", lib_path);
    return false;
  }
  return true;
}

static bool CallAddsOnce(const AddsApi &api, const Tensor &input, Tensor *out,
                         aclrtStream stream, void **workspace,
                         uint64_t *workspace_size) {
  float one = 1.0f;
  float alpha_one = 1.0f;
  aclScalar *other = aclCreateScalar(&one, ACL_FLOAT);
  aclScalar *alpha = aclCreateScalar(&alpha_one, ACL_FLOAT);
  if (other == nullptr || alpha == nullptr) {
    std::fprintf(stderr, "aclCreateScalar failed\n");
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *executor = nullptr;
  auto ret = api.get_ws(input.desc, other, alpha, out->desc, &ws, &executor);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclnnAddsGetWorkspaceSize failed, ret=%d\n",
                 static_cast<int>(ret));
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    return false;
  }
  if (ws > *workspace_size) {
    if (*workspace != nullptr) {
      aclrtFree(*workspace);
      *workspace = nullptr;
    }
    CHECK_ACL(aclrtMalloc(workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    *workspace_size = ws;
  }
  ret = api.run(*workspace, *workspace_size, executor, stream);
  aclDestroyScalar(other);
  aclDestroyScalar(alpha);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclnnAdds failed, ret=%d\n", static_cast<int>(ret));
    return false;
  }
  return true;
}

static bool CallGammaOnce(const GammaApi &api, const Tensor &x1,
                          const Tensor &x2, const Tensor &gamma, double eps,
                          bool add_gamma_offset, Tensor *y, Tensor *rstd,
                          Tensor *x_out, aclrtStream stream, void **workspace,
                          uint64_t *workspace_size) {
  uint64_t ws = 0;
  aclOpExecutor *executor = nullptr;
  auto ret = api.get_ws(x1.desc, x2.desc, gamma.desc, eps, add_gamma_offset,
                        y->desc, rstd->desc, x_out->desc, &ws, &executor);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "GammaAddRmsNorm GetWorkspaceSize failed, ret=%d\n",
                 static_cast<int>(ret));
    return false;
  }
  if (ws > *workspace_size) {
    if (*workspace != nullptr) {
      aclrtFree(*workspace);
      *workspace = nullptr;
    }
    CHECK_ACL(aclrtMalloc(workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    *workspace_size = ws;
  }
  ret = api.run(*workspace, *workspace_size, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "GammaAddRmsNorm run failed, ret=%d\n",
                 static_cast<int>(ret));
    return false;
  }
  return true;
}

static bool CallAddOnce(const AddApi &api, const Tensor &x1, const Tensor &x2,
                        const Tensor &gamma, double eps, Tensor *y,
                        Tensor *rstd, Tensor *x_out, aclrtStream stream,
                        void **workspace, uint64_t *workspace_size) {
  uint64_t ws = 0;
  aclOpExecutor *executor = nullptr;
  auto ret = api.get_ws(x1.desc, x2.desc, gamma.desc, eps, y->desc, rstd->desc,
                        x_out->desc, &ws, &executor);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "AddRmsNorm GetWorkspaceSize failed, ret=%d\n",
                 static_cast<int>(ret));
    return false;
  }
  if (ws > *workspace_size) {
    if (*workspace != nullptr) {
      aclrtFree(*workspace);
      *workspace = nullptr;
    }
    CHECK_ACL(aclrtMalloc(workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    *workspace_size = ws;
  }
  ret = api.run(*workspace, *workspace_size, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "AddRmsNorm run failed, ret=%d\n",
                 static_cast<int>(ret));
    return false;
  }
  return true;
}

static bool CopyToHost(const Tensor &tensor, std::vector<uint8_t> *host) {
  host->assign(tensor.bytes, 0);
  CHECK_ACL(aclrtMemcpy(host->data(), host->size(), tensor.dev, tensor.bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  return true;
}

static bool RunSmoke(const CaseConfig &cfg, const GammaApi &gamma_api,
                     const AddApi &add_api, const AddsApi &adds_api,
                     aclrtStream stream) {
  const int64_t rows = GetEnvI64("X_ROWS", 31);
  const int64_t cols = GetEnvI64("X_COLS", 5120);
  const std::vector<int64_t> x_shape = {rows, cols};
  const std::vector<int64_t> gamma_shape = {cols};
  const std::vector<int64_t> rstd_shape = {rows, 1};

  Tensor x1, x2, gamma_raw, gamma_plus, y_gamma, rstd_gamma, x_gamma;
  Tensor y_add, rstd_add, x_add, y_gamma_raw, rstd_gamma_raw, x_gamma_raw;
  Tensor y_add_raw, rstd_add_raw, x_add_raw;
  auto x1_host = MakePatternData(cfg, x_shape, -1.0f, 1.0f, 17U);
  auto x2_host = MakePatternData(cfg, x_shape, -1.0f, 1.0f, 29U);
  auto gamma_host = MakePatternData(cfg, gamma_shape, -0.5f, 0.5f, 41U);

  bool ok = CreateTensor(x1_host, x_shape, cfg.dtype, cfg.elem_size, &x1) &&
            CreateTensor(x2_host, x_shape, cfg.dtype, cfg.elem_size, &x2) &&
            CreateTensor(gamma_host, gamma_shape, cfg.dtype, cfg.elem_size,
                         &gamma_raw) &&
            CreateZeroTensor(gamma_shape, cfg.dtype, cfg.elem_size,
                             &gamma_plus) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &y_gamma) &&
            CreateZeroTensor(rstd_shape, ACL_FLOAT, sizeof(float), &rstd_gamma) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &x_gamma) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &y_add) &&
            CreateZeroTensor(rstd_shape, ACL_FLOAT, sizeof(float), &rstd_add) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &x_add) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &y_gamma_raw) &&
            CreateZeroTensor(rstd_shape, ACL_FLOAT, sizeof(float),
                             &rstd_gamma_raw) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &x_gamma_raw) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &y_add_raw) &&
            CreateZeroTensor(rstd_shape, ACL_FLOAT, sizeof(float),
                             &rstd_add_raw) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &x_add_raw);

  void *ws_gamma = nullptr;
  void *ws_add = nullptr;
  void *ws_adds = nullptr;
  uint64_t ws_gamma_size = 0;
  uint64_t ws_add_size = 0;
  uint64_t ws_adds_size = 0;
  ok = ok && CallAddsOnce(adds_api, gamma_raw, &gamma_plus, stream, &ws_adds,
                          &ws_adds_size);
  ok = ok && CallGammaOnce(gamma_api, x1, x2, gamma_raw, 1e-6, true,
                           &y_gamma, &rstd_gamma, &x_gamma, stream,
                           &ws_gamma, &ws_gamma_size);
  ok = ok && CallAddOnce(add_api, x1, x2, gamma_plus, 1e-6, &y_add, &rstd_add,
                         &x_add, stream, &ws_add, &ws_add_size);
  ok = ok && CallGammaOnce(gamma_api, x1, x2, gamma_raw, 1e-6, false,
                           &y_gamma_raw, &rstd_gamma_raw, &x_gamma_raw, stream,
                           &ws_gamma, &ws_gamma_size);
  ok = ok && CallAddOnce(add_api, x1, x2, gamma_raw, 1e-6, &y_add_raw,
                         &rstd_add_raw, &x_add_raw, stream, &ws_add,
                         &ws_add_size);
  ok = ok && (aclrtSynchronizeStream(stream) == ACL_SUCCESS);

  std::vector<uint8_t> y_gamma_host;
  std::vector<uint8_t> y_add_host;
  std::vector<uint8_t> x_gamma_host;
  std::vector<uint8_t> x_add_host;
  std::vector<uint8_t> rstd_gamma_host;
  std::vector<uint8_t> rstd_add_host;
  std::vector<uint8_t> gamma_plus_host;
  std::vector<uint8_t> y_gamma_raw_host;
  std::vector<uint8_t> y_add_raw_host;
  std::vector<uint8_t> x_gamma_raw_host;
  std::vector<uint8_t> x_add_raw_host;
  std::vector<uint8_t> rstd_gamma_raw_host;
  std::vector<uint8_t> rstd_add_raw_host;
  ok = ok && CopyToHost(y_gamma, &y_gamma_host) && CopyToHost(y_add, &y_add_host);
  ok = ok && CopyToHost(x_gamma, &x_gamma_host) && CopyToHost(x_add, &x_add_host);
  ok = ok && CopyToHost(rstd_gamma, &rstd_gamma_host) &&
       CopyToHost(rstd_add, &rstd_add_host);
  ok = ok && CopyToHost(gamma_plus, &gamma_plus_host);
  ok = ok && CopyToHost(y_gamma_raw, &y_gamma_raw_host) &&
       CopyToHost(y_add_raw, &y_add_raw_host);
  ok = ok && CopyToHost(x_gamma_raw, &x_gamma_raw_host) &&
       CopyToHost(x_add_raw, &x_add_raw_host);
  ok = ok && CopyToHost(rstd_gamma_raw, &rstd_gamma_raw_host) &&
       CopyToHost(rstd_add_raw, &rstd_add_raw_host);
  size_t mismatch = 0;
  double max_abs = 0.0;
  size_t x_mismatch = 0;
  double x_max_abs = 0.0;
  size_t rstd_mismatch = 0;
  double rstd_max_abs = 0.0;
  size_t raw_mismatch = 0;
  double raw_max_abs = 0.0;
  size_t raw_x_mismatch = 0;
  double raw_x_max_abs = 0.0;
  size_t raw_rstd_mismatch = 0;
  double raw_rstd_max_abs = 0.0;
  size_t gamma_mismatch = 0;
  double gamma_max_abs = 0.0;
  if (ok) {
    auto gamma_plus_ref = MakeGammaPlusData(cfg, gamma_host);
    const size_t gamma_n = static_cast<size_t>(Numel(gamma_shape));
    for (size_t i = 0; i < gamma_n; ++i) {
      double a = ValueAsFloat(cfg, gamma_plus_host, i);
      double b = ValueAsFloat(cfg, gamma_plus_ref, i);
      double diff = std::fabs(a - b);
      gamma_max_abs = std::max(gamma_max_abs, diff);
      if (diff > cfg.atol) {
        ++gamma_mismatch;
      }
    }

    const size_t n = static_cast<size_t>(Numel(x_shape));
    for (size_t i = 0; i < n; ++i) {
      double a = ValueAsFloat(cfg, y_gamma_host, i);
      double b = ValueAsFloat(cfg, y_add_host, i);
      double diff = std::fabs(a - b);
      max_abs = std::max(max_abs, diff);
      if (diff > cfg.atol) {
        ++mismatch;
      }
      double xa = ValueAsFloat(cfg, x_gamma_host, i);
      double xb = ValueAsFloat(cfg, x_add_host, i);
      double xdiff = std::fabs(xa - xb);
      x_max_abs = std::max(x_max_abs, xdiff);
      if (xdiff > cfg.atol) {
        ++x_mismatch;
      }
      double raw_a = ValueAsFloat(cfg, y_gamma_raw_host, i);
      double raw_b = ValueAsFloat(cfg, y_add_raw_host, i);
      double raw_diff = std::fabs(raw_a - raw_b);
      raw_max_abs = std::max(raw_max_abs, raw_diff);
      if (raw_diff > cfg.atol) {
        ++raw_mismatch;
      }
      double raw_xa = ValueAsFloat(cfg, x_gamma_raw_host, i);
      double raw_xb = ValueAsFloat(cfg, x_add_raw_host, i);
      double raw_xdiff = std::fabs(raw_xa - raw_xb);
      raw_x_max_abs = std::max(raw_x_max_abs, raw_xdiff);
      if (raw_xdiff > cfg.atol) {
        ++raw_x_mismatch;
      }
    }
    const size_t rstd_n = static_cast<size_t>(Numel(rstd_shape));
    for (size_t i = 0; i < rstd_n; ++i) {
      double a = GetF32(rstd_gamma_host, i);
      double b = GetF32(rstd_add_host, i);
      double diff = std::fabs(a - b);
      rstd_max_abs = std::max(rstd_max_abs, diff);
      if (diff > 1e-6) {
        ++rstd_mismatch;
      }
      double raw_a = GetF32(rstd_gamma_raw_host, i);
      double raw_b = GetF32(rstd_add_raw_host, i);
      double raw_diff = std::fabs(raw_a - raw_b);
      raw_rstd_max_abs = std::max(raw_rstd_max_abs, raw_diff);
      if (raw_diff > 1e-6) {
        ++raw_rstd_mismatch;
      }
    }
  }

  const bool pass = ok && mismatch == 0 && x_mismatch == 0 &&
                    rstd_mismatch == 0 && raw_mismatch == 0 &&
                    raw_x_mismatch == 0 && raw_rstd_mismatch == 0;
  std::printf("smoke dtype=%s shape=[%ld,%ld] gamma=[%ld] true_max_abs=%.8g "
              "true_mismatch=%zu true_x_max_abs=%.8g true_x_mismatch=%zu "
              "true_rstd_max_abs=%.8g true_rstd_mismatch=%zu "
              "false_max_abs=%.8g false_mismatch=%zu "
              "false_x_max_abs=%.8g false_x_mismatch=%zu "
              "false_rstd_max_abs=%.8g false_rstd_mismatch=%zu "
              "gamma_plus_max_abs=%.8g gamma_plus_mismatch=%zu "
              "status=%s\n",
              cfg.name, rows, cols, cols, max_abs, mismatch, x_max_abs, x_mismatch,
              rstd_max_abs, rstd_mismatch, raw_max_abs, raw_mismatch,
              raw_x_max_abs, raw_x_mismatch, raw_rstd_max_abs,
              raw_rstd_mismatch, gamma_max_abs, gamma_mismatch,
              pass ? "PASS" : "FAIL");

  if (ws_gamma != nullptr) {
    aclrtFree(ws_gamma);
  }
  if (ws_add != nullptr) {
    aclrtFree(ws_add);
  }
  if (ws_adds != nullptr) {
    aclrtFree(ws_adds);
  }
  DestroyTensor(&x1);
  DestroyTensor(&x2);
  DestroyTensor(&gamma_raw);
  DestroyTensor(&gamma_plus);
  DestroyTensor(&y_gamma);
  DestroyTensor(&rstd_gamma);
  DestroyTensor(&x_gamma);
  DestroyTensor(&y_add);
  DestroyTensor(&rstd_add);
  DestroyTensor(&x_add);
  DestroyTensor(&y_gamma_raw);
  DestroyTensor(&rstd_gamma_raw);
  DestroyTensor(&x_gamma_raw);
  DestroyTensor(&y_add_raw);
  DestroyTensor(&rstd_add_raw);
  DestroyTensor(&x_add_raw);
  return pass;
}

static bool TimeGammaOp(const GammaApi &api, const Tensor &x1,
                        const Tensor &x2, const Tensor &gamma,
                        bool add_gamma_offset, Tensor *y, Tensor *rstd,
                        Tensor *x_out, aclrtStream stream, int warmup,
                        int iters, double *avg_us) {
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  for (int i = 0; i < warmup; ++i) {
    if (!CallGammaOnce(api, x1, x2, gamma, 1e-6, add_gamma_offset, y, rstd,
                       x_out, stream, &workspace, &workspace_size)) {
      return false;
    }
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  CHECK_ACL(aclrtCreateEvent(&start));
  CHECK_ACL(aclrtCreateEvent(&end));
  float total_ms = 0.0f;
  for (int i = 0; i < iters; ++i) {
    uint64_t ws = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = api.get_ws(x1.desc, x2.desc, gamma.desc, 1e-6,
                          add_gamma_offset, y->desc, rstd->desc, x_out->desc,
                          &ws, &executor);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "GammaAddRmsNorm GetWorkspaceSize failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      return false;
    }
    if (ws > workspace_size) {
      if (workspace != nullptr) {
        aclrtFree(workspace);
        workspace = nullptr;
      }
      CHECK_ACL(aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST));
      workspace_size = ws;
    }
    CHECK_ACL(aclrtRecordEvent(start, stream));
    ret = api.run(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "GammaAddRmsNorm run failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      return false;
    }
    CHECK_ACL(aclrtRecordEvent(end, stream));
    CHECK_ACL(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsed_ms, start, end));
    total_ms += elapsed_ms;
  }
  *avg_us = static_cast<double>(total_ms) * 1000.0 / static_cast<double>(iters);
  aclrtDestroyEvent(start);
  aclrtDestroyEvent(end);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  return true;
}

static bool TimeAddOp(const AddApi &api, const Tensor &x1, const Tensor &x2,
                      const Tensor &gamma, Tensor *y, Tensor *rstd,
                      Tensor *x_out, aclrtStream stream, int warmup, int iters,
                      double *avg_us) {
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  for (int i = 0; i < warmup; ++i) {
    if (!CallAddOnce(api, x1, x2, gamma, 1e-6, y, rstd, x_out, stream,
                     &workspace, &workspace_size)) {
      return false;
    }
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  CHECK_ACL(aclrtCreateEvent(&start));
  CHECK_ACL(aclrtCreateEvent(&end));
  float total_ms = 0.0f;
  for (int i = 0; i < iters; ++i) {
    uint64_t ws = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = api.get_ws(x1.desc, x2.desc, gamma.desc, 1e-6, y->desc,
                          rstd->desc, x_out->desc, &ws, &executor);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "AddRmsNorm GetWorkspaceSize failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      return false;
    }
    if (ws > workspace_size) {
      if (workspace != nullptr) {
        aclrtFree(workspace);
        workspace = nullptr;
      }
      CHECK_ACL(aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST));
      workspace_size = ws;
    }
    CHECK_ACL(aclrtRecordEvent(start, stream));
    ret = api.run(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "AddRmsNorm run failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      return false;
    }
    CHECK_ACL(aclrtRecordEvent(end, stream));
    CHECK_ACL(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsed_ms, start, end));
    total_ms += elapsed_ms;
  }
  *avg_us = static_cast<double>(total_ms) * 1000.0 / static_cast<double>(iters);
  aclrtDestroyEvent(start);
  aclrtDestroyEvent(end);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  return true;
}

static bool TimeAddsPlusAdd(const AddsApi &adds_api, const AddApi &add_api,
                            const Tensor &x1, const Tensor &x2,
                            const Tensor &gamma_raw, Tensor *gamma_plus,
                            Tensor *y, Tensor *rstd, Tensor *x_out,
                            aclrtStream stream, int warmup, int iters,
                            double *avg_us) {
  void *adds_workspace = nullptr;
  void *add_workspace = nullptr;
  uint64_t adds_workspace_size = 0;
  uint64_t add_workspace_size = 0;
  float one = 1.0f;
  float alpha_one = 1.0f;
  aclScalar *other = aclCreateScalar(&one, ACL_FLOAT);
  aclScalar *alpha = aclCreateScalar(&alpha_one, ACL_FLOAT);
  if (other == nullptr || alpha == nullptr) {
    std::fprintf(stderr, "aclCreateScalar failed in chain timing\n");
    return false;
  }

  auto prepare_adds = [&]() -> aclOpExecutor * {
    uint64_t ws = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = adds_api.get_ws(gamma_raw.desc, other, alpha, gamma_plus->desc,
                               &ws, &executor);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "aclnnAddsGetWorkspaceSize failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      return nullptr;
    }
    if (ws > adds_workspace_size) {
      if (adds_workspace != nullptr) {
        aclrtFree(adds_workspace);
        adds_workspace = nullptr;
      }
      if (aclrtMalloc(&adds_workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMalloc adds workspace failed\n");
        return nullptr;
      }
      adds_workspace_size = ws;
    }
    return executor;
  };

  auto prepare_add = [&]() -> aclOpExecutor * {
    uint64_t ws = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = add_api.get_ws(x1.desc, x2.desc, gamma_plus->desc, 1e-6,
                              y->desc, rstd->desc, x_out->desc, &ws,
                              &executor);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "AddRmsNorm GetWorkspaceSize failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      return nullptr;
    }
    if (ws > add_workspace_size) {
      if (add_workspace != nullptr) {
        aclrtFree(add_workspace);
        add_workspace = nullptr;
      }
      if (aclrtMalloc(&add_workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMalloc add workspace failed\n");
        return nullptr;
      }
      add_workspace_size = ws;
    }
    return executor;
  };

  for (int i = 0; i < warmup; ++i) {
    aclOpExecutor *adds_executor = prepare_adds();
    aclOpExecutor *add_executor = prepare_add();
    if (adds_executor == nullptr || add_executor == nullptr) {
      aclDestroyScalar(other);
      aclDestroyScalar(alpha);
      return false;
    }
    auto ret =
        adds_api.run(adds_workspace, adds_workspace_size, adds_executor, stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "aclnnAdds failed in warmup, ret=%d\n",
                   static_cast<int>(ret));
      aclDestroyScalar(other);
      aclDestroyScalar(alpha);
      return false;
    }
    ret = add_api.run(add_workspace, add_workspace_size, add_executor, stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "AddRmsNorm failed in warmup, ret=%d\n",
                   static_cast<int>(ret));
      aclDestroyScalar(other);
      aclDestroyScalar(alpha);
      return false;
    }
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));

  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  CHECK_ACL(aclrtCreateEvent(&start));
  CHECK_ACL(aclrtCreateEvent(&end));
  float total_ms = 0.0f;
  for (int i = 0; i < iters; ++i) {
    aclOpExecutor *adds_executor = prepare_adds();
    aclOpExecutor *add_executor = prepare_add();
    if (adds_executor == nullptr || add_executor == nullptr) {
      aclDestroyScalar(other);
      aclDestroyScalar(alpha);
      return false;
    }
    CHECK_ACL(aclrtRecordEvent(start, stream));
    auto ret =
        adds_api.run(adds_workspace, adds_workspace_size, adds_executor, stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "aclnnAdds failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      aclDestroyScalar(other);
      aclDestroyScalar(alpha);
      return false;
    }
    ret = add_api.run(add_workspace, add_workspace_size, add_executor, stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "AddRmsNorm failed in timing, ret=%d\n",
                   static_cast<int>(ret));
      aclDestroyScalar(other);
      aclDestroyScalar(alpha);
      return false;
    }
    CHECK_ACL(aclrtRecordEvent(end, stream));
    CHECK_ACL(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsed_ms, start, end));
    total_ms += elapsed_ms;
  }
  *avg_us = static_cast<double>(total_ms) * 1000.0 / static_cast<double>(iters);

  aclrtDestroyEvent(start);
  aclrtDestroyEvent(end);
  aclDestroyScalar(other);
  aclDestroyScalar(alpha);
  if (adds_workspace != nullptr) {
    aclrtFree(adds_workspace);
  }
  if (add_workspace != nullptr) {
    aclrtFree(add_workspace);
  }
  return true;
}

static bool RunPerf(const CaseConfig &cfg, const GammaApi &gamma_api,
                    const AddApi &add_api, const AddsApi &adds_api,
                    aclrtStream stream) {
  const int64_t rows = GetEnvI64("X_ROWS", 31);
  const int64_t cols = GetEnvI64("X_COLS", 5120);
  const std::vector<int64_t> x_shape = {rows, cols};
  const std::vector<int64_t> gamma_shape = {cols};
  const std::vector<int64_t> rstd_shape = {rows, 1};
  Tensor x1, x2, gamma_raw, gamma_plus, y, rstd, x_out;
  auto x1_host = MakeData(cfg, x_shape, 1.0f);
  auto x2_host = MakeData(cfg, x_shape, 0.5f);
  auto gamma_host = MakeData(cfg, gamma_shape, 1.0f);
  auto gamma_plus_host = MakeData(cfg, gamma_shape, 2.0f);
  bool ok = CreateTensor(x1_host, x_shape, cfg.dtype, cfg.elem_size, &x1) &&
            CreateTensor(x2_host, x_shape, cfg.dtype, cfg.elem_size, &x2) &&
            CreateTensor(gamma_host, gamma_shape, cfg.dtype, cfg.elem_size,
                         &gamma_raw) &&
            CreateTensor(gamma_plus_host, gamma_shape, cfg.dtype, cfg.elem_size,
                         &gamma_plus) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &y) &&
            CreateZeroTensor(rstd_shape, ACL_FLOAT, sizeof(float), &rstd) &&
            CreateZeroTensor(x_shape, cfg.dtype, cfg.elem_size, &x_out);

  double gamma_true_us = 0.0;
  double gamma_false_us = 0.0;
  double add_raw_us = 0.0;
  double add_plus_us = 0.0;
  double chain_us = 0.0;
  const int warmup = GetEnvInt("WARMUP", 30);
  const int iters = GetEnvInt("ITERS", 200);
  ok = ok && TimeGammaOp(gamma_api, x1, x2, gamma_raw, true, &y, &rstd,
                         &x_out, stream, warmup, iters, &gamma_true_us);
  ok = ok && TimeGammaOp(gamma_api, x1, x2, gamma_raw, false, &y, &rstd,
                         &x_out, stream, warmup, iters, &gamma_false_us);
  ok = ok && TimeAddOp(add_api, x1, x2, gamma_raw, &y, &rstd, &x_out, stream,
                       warmup, iters, &add_raw_us);
  ok = ok && TimeAddOp(add_api, x1, x2, gamma_plus, &y, &rstd, &x_out, stream,
                       warmup, iters, &add_plus_us);
  ok = ok && TimeAddsPlusAdd(adds_api, add_api, x1, x2, gamma_raw, &gamma_plus,
                             &y, &rstd, &x_out, stream, warmup, iters,
                             &chain_us);
  double true_add_delta = gamma_true_us - add_plus_us;
  double false_add_delta = gamma_false_us - add_raw_us;
  double chain_delta = gamma_true_us - chain_us;
  std::printf("perf dtype=%s shape=[%ld,%ld] gamma_true_us=%.4f "
              "gamma_false_us=%.4f add_raw_us=%.4f add_plus_us=%.4f "
              "chain_us=%.4f true_add_delta_us=%.4f "
              "false_add_delta_us=%.4f chain_delta_us=%.4f "
              "true_add_ratio=%.4f false_add_ratio=%.4f chain_ratio=%.4f "
              "status=%s\n",
              cfg.name, rows, cols, gamma_true_us, gamma_false_us, add_raw_us, add_plus_us,
              chain_us, true_add_delta, false_add_delta, chain_delta,
              add_plus_us > 0.0 ? gamma_true_us / add_plus_us : 0.0,
              add_raw_us > 0.0 ? gamma_false_us / add_raw_us : 0.0,
              chain_us > 0.0 ? gamma_true_us / chain_us : 0.0,
              ok ? "PASS" : "FAIL");

  DestroyTensor(&x1);
  DestroyTensor(&x2);
  DestroyTensor(&gamma_raw);
  DestroyTensor(&gamma_plus);
  DestroyTensor(&y);
  DestroyTensor(&rstd);
  DestroyTensor(&x_out);
  return ok;
}

int main() {
  const char *gamma_lib = std::getenv("GAMMA_OPAPI_LIB");
  if (gamma_lib == nullptr) {
    gamma_lib =
        "/opt/jyf/ops-nn/build_out/_CPack_Packages/Linux/External/"
        "cann-ops-nn-custom_linux-aarch64.run/packages/vendors/custom_nn/"
        "op_api/lib/libcust_opapi.so";
  }
  const char *add_lib = std::getenv("ADD_OPAPI_LIB");
  if (add_lib == nullptr) {
    add_lib =
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/"
        "libopapi_nn.so";
  }
  const char *adds_lib = std::getenv("ADDS_OPAPI_LIB");
  if (adds_lib == nullptr) {
    adds_lib =
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/"
        "libopapi_math.so";
  }

  GammaApi gamma_api;
  AddApi add_api;
  AddsApi adds_api;
  if (!LoadGammaApi(gamma_lib, &gamma_api) ||
      !LoadAddApi(add_lib, &add_api) ||
      !LoadAddsApi(adds_lib, &adds_api)) {
    return 1;
  }

  CHECK_ACL(aclInit(nullptr));
  int32_t device_id = 0;
  CHECK_ACL(aclrtSetDevice(device_id));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  std::vector<CaseConfig> cases = {
      {"fp16", ACL_FLOAT16, 2, 0.0},
      {"bf16", ACL_BF16, 2, 0.0},
      {"fp32", ACL_FLOAT, 4, 1e-5},
  };

  bool all_ok = true;
  const char *case_filter = std::getenv("CASE_FILTER");
  const bool run_smoke = EnvEnabled("RUN_SMOKE", true);
  for (const auto &cfg : cases) {
    if (case_filter != nullptr && case_filter[0] != '\0' &&
        std::strcmp(case_filter, cfg.name) != 0) {
      continue;
    }
    if (run_smoke) {
      all_ok = RunSmoke(cfg, gamma_api, add_api, adds_api, stream) && all_ok;
    }
    all_ok = RunPerf(cfg, gamma_api, add_api, adds_api, stream) && all_ok;
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(device_id);
  aclFinalize();
  return all_ok ? 0 : 2;
}
