#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <optional>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Scoped owners for CUDA resources, and the one place a CUDA status is checked.
namespace cuda {

inline void check(cudaError_t status, std::source_location where = std::source_location::current()) {
  if (status != cudaSuccess)
    throw std::runtime_error{std::string{"CUDA error at "} + where.file_name() + ":" +
                             std::to_string(where.line()) + ": " + cudaGetErrorString(status)};
}

// Device memory for `size()` elements of T, freed on destruction.
template <class T>
class DeviceArray {
public:
  DeviceArray() = default;

  explicit DeviceArray(std::size_t size)
      : size_{size} {
    if (size_ > 0)
      check(cudaMalloc(&data_, size_ * sizeof(T)));
  }

  // Allocates and copies `values` to the device.
  explicit DeviceArray(std::span<const T> values)
      : DeviceArray{values.size()} {
    upload(values);
  }

  // nullopt if the device is out of memory, so the caller can fall back.
  [[nodiscard]] static std::optional<DeviceArray> try_allocate(std::size_t size) {
    DeviceArray array;
    const auto status{cudaMalloc(&array.data_, size * sizeof(T))};
    if (status == cudaErrorMemoryAllocation) {
      cudaGetLastError(); // clear the sticky error
      return std::nullopt;
    }
    check(status);
    array.size_ = size;
    return array;
  }

  ~DeviceArray() { cudaFree(data_); }

  DeviceArray(DeviceArray&& other) noexcept
      : data_{std::exchange(other.data_, nullptr)}
      , size_{std::exchange(other.size_, 0)} {}

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void upload(std::span<const T> values) {
    if (!values.empty())
      check(cudaMemcpy(data_, values.data(), values.size_bytes(), cudaMemcpyHostToDevice));
  }

  void zero() { check(cudaMemset(data_, 0, size_ * sizeof(T))); }

  // The first `count` elements.
  [[nodiscard]] std::vector<T> download(std::size_t count) const {
    std::vector<T> values(count);
    if (count > 0)
      check(cudaMemcpy(values.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }

private:
  T* data_{nullptr};
  std::size_t size_{0};
};

// Times the device work queued between construction and stop_seconds().
class Timer {
public:
  Timer() {
    check(cudaEventCreate(&start_));
    check(cudaEventCreate(&stop_));
    check(cudaEventRecord(start_));
  }

  ~Timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  // Waits for the queued work to finish.
  [[nodiscard]] double stop_seconds() {
    check(cudaEventRecord(stop_));
    check(cudaEventSynchronize(stop_));
    float milliseconds{0.0f};
    check(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds / 1000.0;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

} // namespace cuda
