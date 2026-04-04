#pragma once

#include <string>
#include <vector>

#include "novaphy/core/aabb.h"

namespace novaphy {

struct BroadPhasePair;

/**
 * @brief Abstract broadphase interface for pluggable broadphase algorithms.
 *
 * @details Any broadphase implementation (SAP, grid, BVH) can implement
 * this interface and be used interchangeably by the collision pipeline
 * and solvers.
 */
class IBroadPhase {
public:
    virtual ~IBroadPhase() = default;

    /**
     * @brief Update candidate overlap pairs from current world-space AABBs.
     *
     * @param [in] aabbs World-space axis-aligned bounding boxes.
     * @param [in] static_mask Static-body mask (skip static-static pairs).
     */
    virtual void update(const std::vector<AABB>& aabbs,
                        const std::vector<bool>& static_mask) = 0;

    /**
     * @brief Get broadphase overlap candidates from the latest update.
     *
     * @return Read-only list of potentially overlapping pairs.
     */
    virtual const std::vector<BroadPhasePair>& get_pairs() const = 0;

    /**
     * @brief Human-readable name of this broadphase algorithm.
     *
     * @return Name string.
     */
    virtual std::string name() const = 0;
};

}  // namespace novaphy
