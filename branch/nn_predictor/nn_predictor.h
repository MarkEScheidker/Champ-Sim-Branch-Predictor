#ifndef NN_PREDICTOR_H
#define NN_PREDICTOR_H

#include <cstdint>
#include <deque>
#include <tensorflow/c/c_api.h>

#include "modules.h"

//neural Network Branch Predictor using TensorFlow
struct nn_predictor : public champsim::modules::branch_predictor {
  using branch_predictor::branch_predictor;

  static constexpr std::size_t HISTORY_SIZE = 64;

  //tensorFlow state
  TF_Graph* graph = nullptr;
  TF_Status* status = nullptr;
  TF_Session* session = nullptr;
  TF_Operation* input_op = nullptr;
  TF_Operation* output_op = nullptr;
  TF_Tensor* reusable_input = nullptr;

  //history of recent branch outcomes
  std::deque<bool> branch_history;

  explicit nn_predictor(O3_CPU* cpu);
  ~nn_predictor() = default;  //resources managed elsewhere

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif // NN_PREDICTOR_H
