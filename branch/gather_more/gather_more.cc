#include "gather_more.h"
#include <iomanip>

gather_more::gather_more(O3_CPU* cpu)
  : champsim::modules::branch_predictor(cpu),
    data_file("gather_more.csv")
{
    if (data_file.is_open()) {
        // header: pc_i,target_i,type_i,taken_i  (i = 0..63) then "new"
        for (std::size_t i = 0; i < HISTORY_SIZE; ++i) {
            data_file << "pc"    << i << ','
                      << "target"<< i << ','
                      << "type"  << i << ','
                      << "taken" << i << ',';
        }
        data_file << "new\n";
    }
}

bool gather_more::predict_branch(champsim::address /*ip*/)
{
    return true;   // still an always-taken stub
}

void gather_more::last_branch_result(champsim::address ip,
                                     champsim::address branch_target,
                                     bool taken,
                                     uint8_t branch_type)
{
    ++branch_count;

    //low 8 bits of PC and target
    auto pc_low8     = static_cast<uint8_t>(ip.to<uint64_t>() & 0xFFu);
    auto target_low8 = static_cast<uint8_t>(branch_target.to<uint64_t>() & 0xFFu);

    branch_history.push_back({ pc_low8, target_low8, branch_type, taken });
    if (branch_history.size() > HISTORY_SIZE)
        branch_history.pop_front();

    //store only every 100th sample
    if (branch_history.size() == HISTORY_SIZE &&
        (branch_count % 100) == 0 &&
        data_file.is_open())
    {
        for (const auto& bf : branch_history) {
            data_file << +bf.pc_low8   << ','
                      << +bf.target_low8 << ','
                      << +bf.branch_type << ','
                      << (bf.taken ? 1 : 0) << ',';
        }
        data_file << (taken ? 1 : 0) << '\n';
    }
}
