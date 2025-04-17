#include "nn_predictor.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

// helper function for TensorFlow to deallocate memory
namespace {
static void free_deallocator(void* data, std::size_t, void*) {
  std::free(data);
}
}  // namespace

// Constructor: Load TensorFlow model and allocate input tensor
nn_predictor::nn_predictor(O3_CPU* cpu) : branch_predictor(cpu) {
  status = TF_NewStatus();
  graph = TF_NewGraph();
  TF_SessionOptions* opts = TF_NewSessionOptions();

  const char* model_dir = "branch/nn_predictor/model";
  const char* input_name = "serve_keras_tensor";
  const char* output_name = "StatefulPartitionedCall";
  const char* tags[] = {"serve"};

  session = TF_LoadSessionFromSavedModel(opts, nullptr, model_dir, tags, 1, graph, nullptr, status);
  if (TF_GetCode(status) != TF_OK) {
    session = nullptr;
    return;
  }

  input_op = TF_GraphOperationByName(graph, input_name);
  output_op = TF_GraphOperationByName(graph, output_name);

  if (!session || !input_op || !output_op) {
    return;
  }

  //allocate reusable input tensor
  int64_t dims[2] = {1, static_cast<int64_t>(HISTORY_SIZE)};
  void* buf = std::calloc(HISTORY_SIZE, sizeof(float));
  reusable_input = TF_NewTensor(TF_FLOAT, dims, 2, buf, HISTORY_SIZE * sizeof(float), free_deallocator, nullptr);
}

// predict branch outcome using TensorFlow model
bool nn_predictor::predict_branch(champsim::address /*ip*/) {
  if (!session || !input_op || !output_op || !reusable_input) {
    return true;  // default to taken if tensorflow isn't ready
  }

  //prepare input tensor from branch history
  float* buf = static_cast<float*>(TF_TensorData(reusable_input));
  if (!buf) {
    return true;
  }

  //zero-padding if branch history is shorter than HISTORY_SIZE
  std::size_t pad = (branch_history.size() < HISTORY_SIZE) ? (HISTORY_SIZE - branch_history.size()) : 0;
  std::size_t i = 0;
  for (; i < pad; ++i)
    buf[i] = 0.f;
  for (bool b : branch_history)
    buf[i++] = b ? 1.f : 0.f;

  //run tensorflow inference
  TF_Output in = {input_op, 0};
  TF_Output out = {output_op, 0};
  TF_Tensor* out_tensor = nullptr;

  TF_SessionRun(session, nullptr, &in, &reusable_input, 1, &out, &out_tensor, 1, nullptr, 0, nullptr, status);

  if (TF_GetCode(status) != TF_OK || !out_tensor) {
    if (out_tensor)
      TF_DeleteTensor(out_tensor);
    return true;
  }

  float prob = *static_cast<const float*>(TF_TensorData(out_tensor));
  TF_DeleteTensor(out_tensor);

  return prob >= 0.5f;  // predict taken if probability is more than 50 percent
}

// Update branch history with actual outcome
void nn_predictor::last_branch_result(champsim::address, champsim::address, bool taken, uint8_t) {
  branch_history.push_back(taken);
  if (branch_history.size() > HISTORY_SIZE)
    branch_history.pop_front();
}
