#ifndef BRANCH_GATHER_BASIC_H
#define BRANCH_GATHER_BASIC_H

#include <deque>
#include <fstream>
#include <cstdint>
#include "modules.h"

// Minimal branch predictor that always predicts taken and logs every branch.
// Each log entry is a flattened vector of the last 64 branch outcomes (padded if needed)
// followed by the current branch outcome.
struct gather_basic : champsim::modules::branch_predictor {
  using branch_predictor::branch_predictor; // Inherit base constructor

  static constexpr std::size_t HISTORY_SIZE = 64;

  // Holds the most recent branch outcomes (true for taken, false for not taken)
  std::deque<bool> branch_history;

  // CSV output file stream
  std::ofstream data_file;

  // Branch counter (for potential timestamping, if desired)
  std::uint64_t branch_count = 0;

  // Always predict taken.
  bool predict_branch(champsim::address ip);

  // Called after the branch outcome is known.
  // Logs a feature vector for every branch: the last 64 outcomes (padded if needed) and the current outcome.
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);

  // Constructor taking the CPU pointer.
  explicit gather_basic(O3_CPU* cpu);
};

#endif
