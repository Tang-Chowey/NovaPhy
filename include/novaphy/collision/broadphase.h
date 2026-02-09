#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "novaphy/core/aabb.h"

namespace novaphy {

/// A pair of potentially overlapping bodies
struct BroadPhasePair {
    int body_a;
    int body_b;

    bool operator==(const BroadPhasePair& other) const {
        return body_a == other.body_a && body_b == other.body_b;
    }
};

/// Sweep and Prune broadphase collision detection.
/// Maintains sorted endpoint lists along each axis and uses insertion sort
/// to exploit temporal coherence between frames.
class SweepAndPrune {
public:
    SweepAndPrune() = default;

    /// Update broadphase with current AABBs for all bodies.
    /// body_aabbs[i] is the world-space AABB for body i.
    /// static_mask[i] is true if body i is static (skip static-static pairs).
    void update(const std::vector<AABB>& body_aabbs,
                const std::vector<bool>& static_mask);

    /// Get the list of potentially overlapping pairs
    const std::vector<BroadPhasePair>& get_pairs() const { return pairs_; }

private:
    struct Endpoint {
        float value;
        int body_index;
        bool is_min;  // true = min endpoint, false = max endpoint
    };

    std::vector<Endpoint> endpoints_x_;
    std::vector<Endpoint> endpoints_y_;
    std::vector<Endpoint> endpoints_z_;
    std::vector<BroadPhasePair> pairs_;

    bool initialized_ = false;

    void rebuild(const std::vector<AABB>& aabbs, const std::vector<bool>& static_mask);
    static void insertion_sort(std::vector<Endpoint>& eps);
};

}  // namespace novaphy
