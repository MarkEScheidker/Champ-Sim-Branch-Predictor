#ifndef BRANCH_GATHER_MORE_H
#define BRANCH_GATHER_MORE_H

#include <deque>
#include <fstream>
#include <cstdint>
#include "modules.h"

//one per-branch feature record
struct BranchFeature {
    std::uint8_t  pc_low8;       // pc & 0xFF
    std::uint8_t  target_low8;   // target & 0xFF
    uint8_t       branch_type;
    bool          taken;
};

struct gather_more : champsim::modules::branch_predictor {
    using branch_predictor::branch_predictor;

    static constexpr std::size_t HISTORY_SIZE = 64;

    explicit gather_more(O3_CPU* cpu);
    bool predict_branch(champsim::address ip);
    void last_branch_result(champsim::address ip,
                            champsim::address branch_target,
                            bool taken,
                            uint8_t branch_type);

private:
    std::deque<BranchFeature> branch_history;
    std::ofstream             data_file;
    std::uint64_t             branch_count = 0;
};

#endif // BRANCH_GATHER_MORE_H
