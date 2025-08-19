#include "timer.h"
#include <ctime>
#include <thread>

using namespace std::chrono_literals;

Timer::Timer(const bool &auto_start) {
  if (auto_start) {
    start();
  }

  return;
}

const bool Timer::reset(const bool &auto_start) {
  start_time_ = -1;
  time_sum_ = 0;

  if (auto_start) {
    start();
  }

  return true;
}

const bool Timer::start() {
  if (start_time_ >= 0) {
    return true;
  }

  start_time_ = clock();

  return true;
}

const bool Timer::pause() {
  if (start_time_ < 0) {
    return true;
  }

  time_sum_ += (clock() - start_time_) / CLOCKS_PER_SEC;
  start_time_ = -1;

  return true;
}

const double Timer::currentTimeSum() {
  if (start_time_ < 0) {
    return time_sum_;
  }

  return (clock() - start_time_) / CLOCKS_PER_SEC + time_sum_;
}

const double Timer::now(const std::string &append) {
  const double current_time_sum = currentTimeSum();

  if (append == "s") {
    return current_time_sum;
  }
  if (append == "m") {
    return current_time_sum / 60.0;
  }
  if (append == "h") {
    return current_time_sum / 60.0 / 60.0;
  }
  if (append == "d") {
    return current_time_sum / 60.0 / 60.0 / 24.0;
  }

  return current_time_sum;
}

const bool Timer::sleep(const long int &second) {
  std::this_thread::sleep_for(std::chrono::seconds(second));

  return true;
}
