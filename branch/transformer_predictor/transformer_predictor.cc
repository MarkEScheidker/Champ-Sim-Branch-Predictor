#include "transformer_predictor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace
{
static void free_deallocator(void* data, std::size_t, void*) { std::free(data); }

static std::vector<uint8_t> top4(const std::vector<uint8_t>& values)
{
  std::unordered_map<uint8_t, int> cnt;   //frequency so far
  std::unordered_map<uint8_t, int> first; //first index seen
  std::vector<uint8_t> top;               //current top list

  for (int i = 0; i < (int)values.size(); ++i) {
    uint8_t v = values[i];
    if (!cnt.count(v))
      first[v] = i; //remember first position
    ++cnt[v];

    if (std::find(top.begin(), top.end(), v) == top.end()) {
      if ((int)top.size() < 4) {
        top.push_back(v);
      } else {
        //find worst = lowest count, tie-break by earliest first index
        auto worst_it = std::min_element(top.begin(), top.end(), [&](uint8_t a, uint8_t b) {
          if (cnt[a] != cnt[b])
            return cnt[a] < cnt[b];
          return first[a] < first[b];
        });
        uint8_t worst = *worst_it;
        if ((cnt[v] > cnt[worst]) || (cnt[v] == cnt[worst] && first[v] > first[worst])) {
          *worst_it = v;
        }
      }
    }
  }
  return top; //size <= 4, order preserved from algo
}
} // namespace

transformer_predictor::transformer_predictor(O3_CPU* cpu) : branch_predictor(cpu)
{
  status = TF_NewStatus();
  graph = TF_NewGraph();
  TF_SessionOptions* opts = TF_NewSessionOptions();

  const char* model_dir = "branch/transformer_predictor/model";
  const char* input_name = "serving_default_bits";
  const char* output_name = "StatefulPartitionedCall";
  const char* tags[] = {"serve"};

  session = TF_LoadSessionFromSavedModel(opts, nullptr, model_dir, tags, 1, graph, nullptr, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "load session failed: " << TF_Message(status) << "\n";
    session = nullptr;
    return;
  }

  input_op = TF_GraphOperationByName(graph, input_name);
  output_op = TF_GraphOperationByName(graph, output_name);
  if (!input_op)
    std::cerr << "input op not found\n";
  if (!output_op)
    std::cerr << "output op not found\n";
  if (!session || !input_op || !output_op)
    return;

  int64_t dims[3] = {1, static_cast<int64_t>(HISTORY_SIZE), 12}; //(1,64,12)
  std::size_t total = HISTORY_SIZE * 12;
  void* buf = std::calloc(total, sizeof(float));
  reusable_input = TF_NewTensor(TF_FLOAT, dims, 3, buf, total * sizeof(float), free_deallocator, nullptr);
}

bool transformer_predictor::predict_branch(champsim::address)
{
  if (!session || !input_op || !output_op || !reusable_input)
    return true;

  float* buf = static_cast<float*>(TF_TensorData(reusable_input));
  if (!buf)
    return true;

  //build helper vectors for top4
  std::vector<uint8_t> pc_vals, tgt_vals;
  pc_vals.reserve(branch_history.size());
  tgt_vals.reserve(branch_history.size());
  for (auto& e : branch_history) {
    pc_vals.push_back(e.pc_low8);
    tgt_vals.push_back(e.target_low8);
  }
  auto top_pc = top4(pc_vals);
  auto top_targ = top4(tgt_vals);

  //pad older missing entries with zeros
  std::size_t pad = branch_history.size() < HISTORY_SIZE ? (HISTORY_SIZE - branch_history.size()) : 0;
  std::size_t idx = 0;
  std::memset(buf, 0, pad * 12 * sizeof(float));
  idx += pad * 12;

  //encode actual history
  for (std::size_t h = 0; h < branch_history.size(); ++h) {
    const auto& e = branch_history[h];

    //taken
    buf[idx++] = e.taken ? 1.0f : 0.0f;

    //PC one-hot among top_pc
    for (int i = 0; i < 4; ++i)
      buf[idx++] = (i < (int)top_pc.size() && e.pc_low8 == top_pc[i]) ? 1.0f : 0.0f;

    //target one-hot among top_targ
    for (int i = 0; i < 4; ++i)
      buf[idx++] = (i < (int)top_targ.size() && e.target_low8 == top_targ[i]) ? 1.0f : 0.0f;

    //branch type one-hot 3 bits
    buf[idx++] = (e.branch_type == 0) ? 1.0f : 0.0f;
    buf[idx++] = (e.branch_type == 1) ? 1.0f : 0.0f;
    buf[idx++] = (e.branch_type == 2) ? 1.0f : 0.0f;
  }

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
  return prob >= 0.5f;
}

void transformer_predictor::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  uint8_t pc_lo = static_cast<uint8_t>(ip.to<uint64_t>() & 0xFFu);
  uint8_t tgt_lo = static_cast<uint8_t>(branch_target.to<uint64_t>() & 0xFFu);
  branch_history.push_back({pc_lo, tgt_lo, branch_type, taken});
  if (branch_history.size() > HISTORY_SIZE)
    branch_history.pop_front();
}
