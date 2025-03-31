#include "gather_basic.h"
#include <iomanip>

// Constructor: open the CSV file and write header.
gather_basic::gather_basic(O3_CPU* cpu)
  : champsim::modules::branch_predictor(cpu)
{
  data_file.open("gather_basic.csv");
  if (data_file.is_open()) {
    // Header: b0,b1,...,b63,new
    for (std::size_t i = 0; i < HISTORY_SIZE; i++) {
      data_file << "b" << i << ",";
    }
    data_file << "new" << "\n";
  }
}

// Always predict taken.
bool gather_basic::predict_branch(champsim::address /*ip*/)
{
  return true;
}

void gather_basic::last_branch_result(champsim::address ip, champsim::address /*branch_target*/, bool taken, uint8_t /*branch_type*/)
{
  branch_count++;

  // Create a temporary vector of size HISTORY_SIZE:
  // If we have fewer than 64 outcomes, pad with zeros at the beginning.
  std::deque<bool> full_history;
  std::size_t pad = (branch_history.size() < HISTORY_SIZE) ? HISTORY_SIZE - branch_history.size() : 0;
  for (std::size_t i = 0; i < pad; i++) {
    full_history.push_back(false);
  }
  for (bool b : branch_history) {
    full_history.push_back(b);
  }

  // Write the full_history vector (exactly HISTORY_SIZE values) and then the current outcome.
  if (data_file.is_open()) {
    for (std::size_t i = 0; i < HISTORY_SIZE; i++) {
      data_file << (full_history[i] ? 1 : 0) << ",";
    }
    data_file << (taken ? 1 : 0) << "\n";
  }

  // Update branch history:
  branch_history.push_back(taken);
  if (branch_history.size() > HISTORY_SIZE) {
    branch_history.pop_front();
  }
}
