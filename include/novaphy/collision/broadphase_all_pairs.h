#pragma once

#include <vector>

#include "novaphy/collision/ibroadphase.h"
#include "novaphy/collision/broadphase.h"
#include "novaphy/core/aabb.h"

namespace novaphy {

/**
 * @brief Brute-force O(n^2) broadphase that reports all non-static pairs.
 *
 * @details Useful for debugging, validation, and very small scenes.
 * Every non-static pair with overlapping AABBs is reported.
 */
class BroadPhaseAllPairs : public IBroadPhase {
public:
    void update(const std::vector<AABB>& aabbs,
                const std::vector<bool>& static_mask) override {
        pairs_.clear();
        const int n = static_cast<int>(aabbs.size());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (static_mask[i] && static_mask[j]) continue;
                if (aabbs[i].overlaps(aabbs[j])) {
                    pairs_.push_back({i, j});
                }
            }
        }
    }

    const std::vector<BroadPhasePair>& get_pairs() const override { return pairs_; }

    std::string name() const override { return "BroadPhaseAllPairs"; }

private:
    std::vector<BroadPhasePair> pairs_;
};

}  // namespace novaphy
