#pragma once

#include <vector>

#include "novaphy/math/math_types.h"

namespace novaphy {

/**
 * @brief Supported joint-drive target modes.
 */
enum class JointTargetMode {
    Off = 0,            /**< Drive inactive. */
    TargetPosition,     /**< PD drive toward a target position. */
    TargetVelocity,     /**< PD drive toward a target velocity. */
};
/**
 * @brief Per-joint drive configuration.
 *
 * @details Defines a PD-style joint actuator that can target either a
 * position or a velocity. The resulting joint force is clamped to
 * `force_limit`.
 */
struct JointDrive {
    JointTargetMode mode = JointTargetMode::Off;  /**< Active drive mode. */
    float target_position = 0.0f;   /**< Target generalized position. */
    float target_velocity = 0.0f;   /**< Target generalized velocity. */
    float stiffness = 0.0f;         /**< Proportional gain (N·m/rad or N/m). */
    float damping = 0.0f;           /**< Derivative gain (N·m·s/rad or N·s/m). */
    float force_limit = 1e6f;       /**< Maximum drive force/torque magnitude. */
};

/**
 * @brief Unified runtime control input for one simulation step.
 *
 * @details Collects all external inputs that drive the simulation forward:
 * feedforward joint forces, per-joint PD drives, and per-body external
 * wrenches. The solver reads this struct at each `step()` call.
 *
 * Array sizes must match the corresponding model dimensions (joint DOFs,
 * number of links/bodies). An empty vector means "no input for this channel".
 */
struct Control {
    VecXf joint_forces;                   /**< Flat feedforward joint torques/forces. For multi-articulation worlds, this may be the concatenation of all articulation DOFs in model order. */
    std::vector<VecXf> articulation_joint_forces; /**< Per-articulation feedforward torques/forces, one VecXf per articulation. Takes precedence over `joint_forces` when provided. */
    std::vector<JointDrive> joint_drives; /**< Per-joint PD drive configuration, length = num_joints. */

    std::vector<Vec3f> body_forces;       /**< External world-frame forces at body CoM (N). */
    std::vector<Vec3f> body_torques;      /**< External world-frame torques on bodies (N·m). */
};

}  // namespace novaphy
