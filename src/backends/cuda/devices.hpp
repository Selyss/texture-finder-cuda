#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "backends/cuda/device.hpp"

// The devices a search runs on. A search with one device runs on the calling
// thread as it always has. With several, each device gets a host thread of
// its own, because a CUDA device is selected per thread.
namespace cuda {

namespace detail {

struct DeviceState {
  std::mutex mutex;
  std::vector<int> ids;                 // empty until chosen or first used
  bool chosen{false};                   // by choose_devices, not by default
  bool started{false};                  // a call has already used every device
  double rate[2]{0.0, 0.0};             // positions per second on one device, per Path
  std::vector<std::uint64_t> positions; // searched by each entry of ids
};

inline DeviceState& device_state() {
  static DeviceState state;
  return state;
}

[[nodiscard]] inline int device_count() {
  int count{0};
  check(cudaGetDeviceCount(&count));
  if (count < 1)
    throw std::runtime_error{"no CUDA device available"};
  return count;
}

} // namespace detail

// Restricts the search to these device indices, in this order. An index may
// repeat, which gives that device several workers: the validation job does
// this to run the multi-device paths on one GPU.
inline void choose_devices(std::span<const int> ids) {
  const auto count{detail::device_count()};
  if (ids.empty())
    throw std::invalid_argument{"no device given"};
  for (const int id : ids) {
    if (id < 0 || id >= count)
      throw std::invalid_argument{"no CUDA device " + std::to_string(id) + " (this machine has " +
                                  std::to_string(count) + ")"};
  }
  auto& state{detail::device_state()};
  const std::scoped_lock lock{state.mutex};
  state.ids.assign(ids.begin(), ids.end());
  state.positions.assign(ids.size(), 0);
  state.chosen = true;
}

// The chosen devices, or every device of the machine if none were chosen.
[[nodiscard]] inline std::vector<int> devices() {
  auto& state{detail::device_state()};
  const std::scoped_lock lock{state.mutex};
  if (state.ids.empty()) {
    state.ids.resize(static_cast<std::size_t>(detail::device_count()));
    std::iota(state.ids.begin(), state.ids.end(), 0);
    state.positions.assign(state.ids.size(), 0);
  }
  return state.ids;
}

// The two ways the backend searches, which run at very different rates.
enum class Path { bitplanes, standard };

// Starting work on another device costs a few tenths of a second, so a call
// that one device would finish sooner than this stays on the first device.
constexpr double MIN_SHARED_SECONDS{1.0};

// Records how fast one device searched, to judge later calls by.
inline void note_rate(Path path, std::uint64_t positions, double seconds, std::size_t device_count) {
  if (seconds <= 0 || positions == 0)
    return;
  auto& state{detail::device_state()};
  const std::scoped_lock lock{state.mutex};
  state.rate[static_cast<std::size_t>(path)] =
      static_cast<double>(positions) / (seconds * static_cast<double>(device_count));
}

// The devices that a call searching `positions` positions should run on. A
// list given with choose_devices is used as it is. Otherwise every device is
// used once a call is long enough to repay starting them, judged from the
// rate of earlier calls: the session's first call is always a short one.
[[nodiscard]] inline std::vector<int> devices_for(Path path, std::uint64_t positions) {
  auto ids{devices()};
  auto& state{detail::device_state()};
  const std::scoped_lock lock{state.mutex};
  if (state.chosen || state.started)
    return ids;
  const auto rate{state.rate[static_cast<std::size_t>(path)]};
  if (rate > 0 && static_cast<double>(positions) / rate >= MIN_SHARED_SECONDS)
    state.started = true; // the other devices are warm from here on
  else
    ids.resize(1);
  return ids;
}

// "2 x A100-SXM4-40GB, 1 x RTX 3090" for the devices in use.
[[nodiscard]] inline std::string describe_devices() {
  std::vector<std::string> names;
  std::map<std::string, int> counts;
  for (const int id : devices()) {
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, id));
    if (++counts[properties.name] == 1)
      names.emplace_back(properties.name);
  }
  std::string text;
  for (const auto& name : names) {
    if (!text.empty())
      text += ", ";
    text += std::to_string(counts[name]);
    text += " x ";
    text += name;
  }
  return text;
}

// Adds to the tally of positions that worker `index` has searched.
inline void add_work(std::size_t index, std::uint64_t positions) {
  auto& state{detail::device_state()};
  const std::scoped_lock lock{state.mutex};
  if (index < state.positions.size())
    state.positions[index] += positions;
}

[[nodiscard]] inline std::vector<std::uint64_t> work_done() {
  auto& state{detail::device_state()};
  const std::scoped_lock lock{state.mutex};
  return state.positions;
}

// Runs task(index) once per entry of `ids`, each on its own thread with that
// device selected, and waits for all of them. Rethrows the first failure.
template <class Task>
void on_each_device(std::span<const int> ids, Task&& task) {
  std::vector<std::exception_ptr> failures(ids.size());
  std::vector<std::thread> threads;
  threads.reserve(ids.size());
  for (std::size_t index{0}; index < ids.size(); ++index) {
    threads.emplace_back([&, index] {
      try {
        check(cudaSetDevice(ids[index]));
        task(index);
      } catch (...) {
        failures[index] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads)
    thread.join();
  for (const auto& failure : failures) {
    if (failure)
      std::rethrow_exception(failure);
  }
}

} // namespace cuda
