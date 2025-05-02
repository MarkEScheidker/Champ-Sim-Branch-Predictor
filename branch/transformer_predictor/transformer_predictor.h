#ifndef TRANSFORMER_PREDICTOR_H
#define TRANSFORMER_PREDICTOR_H

#include <cstdint>
#include <deque>
#include <tensorflow/c/c_api.h>

#include "modules.h"

struct transformer_predictor : champsim::modules::branch_predictor {
  using branch_predictor::branch_predictor;

  static constexpr std::size_t HISTORY_SIZE = 64;

  //tensorflow state
  TF_Graph* graph = nullptr;
  TF_Status* status = nullptr;
  TF_Session* session = nullptr;
  TF_Operation* input_op = nullptr;
  TF_Operation* output_op = nullptr;
  TF_Tensor* reusable_input = nullptr;

  //branch history entry structure
  struct HistoryEntry {
    std::uint8_t pc_low8;
    std::uint8_t target_low8;
    std::uint8_t branch_type;
    bool taken;
  };
  std::deque<HistoryEntry> branch_history;

  explicit transformer_predictor(O3_CPU* cpu);
  ~transformer_predictor() = default;

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif // TRANSFORMER_PREDICTOR_H
