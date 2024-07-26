#pragma once

#include <string>

class Timer {
public:
  Timer(const bool &auto_start = true);

  const bool reset(const bool &auto_start = true);

  const bool start();

  const bool pause();

  const double currentTimeSum();

  const double now(const std::string &append = "s");

  const bool sleep(const long int &second);

private:
  double start_time_ = -1.0f;
  double time_sum_ = 0.0f;
};
