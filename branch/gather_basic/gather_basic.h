#ifndef BRANCH_GATHER_BASIC_H
#define BRANCH_GATHER_BASIC_H

#include <deque>
#include <fstream>
#include <cstdint>
#include "modules.h"

struct gather_basic : champsim::modules::branch_predictor {
  using branch_predictor::branch_predictor;

  static constexpr std::size_t HISTORY_SIZE = 64;
  std::deque<bool> branch_history;

  std::ofstream data_file;

  std::uint64_t branch_count = 0;

  bool predict_branch(champsim::address ip);

  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
  
  explicit gather_basic(O3_CPU* cpu);
};

#endif
