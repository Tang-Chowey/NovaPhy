/**
 * @file multibody.cpp
 * @brief Stateful multibody dynamics backed by Featherstone algorithms.
 *
 * @details Forward dynamics delegates to featherstone::forward_dynamics()
 * (CRBA + Cholesky). The constraint-solver support methods (calc_acceleration_deltas,
 * fill_constraint_jacobian) use the cached FK result and factored mass matrix
 * produced by compute_accelerations_aba().
 */
#include "novaphy/multibody/multibody.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace novaphy {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MultiBody::MultiBody(const Articulation& model,
                     float linear_damping, float angular_damping)
    : model_(model), linear_damping_(linear_damping), angular_damping_(angular_damping) {
    const int n = model.num_links();
    const int nq = model.total_q();
    const int nv = model.total_qd();

    q_ = VecXf::Zero(nq);
    qd_ = VecXf::Zero(nv);
    tau_ = VecXf::Zero(nv);

    link_forces_.resize(n, Vec3f::Zero());
    link_torques_.resize(n, Vec3f::Zero());
    constraint_forces_.resize(n, Vec3f::Zero());
    constraint_torques_.resize(n, Vec3f::Zero());

    world_transforms_.resize(n);
    output_ = VecXf::Zero(6 + nv);

    // Pre-compute DOF offsets
    dof_offsets_.resize(n);
    cfg_offsets_.resize(n);
    int dof_off = 0, cfg_off = 0;
    for (int i = 0; i < n; ++i) {
        dof_offsets_[i] = dof_off;
        cfg_offsets_[i] = cfg_off;
        dof_off += model.joints[i].num_qd();
        cfg_off += model.joints[i].num_q();
    }

    // Initialize mass matrix cache with identity (sized, but will be recomputed)
    mass_matrix_cache_ = MatXf::Identity(nv, nv);
    mass_llt_ = mass_matrix_cache_.llt();
}

// ---------------------------------------------------------------------------
// Force accumulation
// ---------------------------------------------------------------------------

void MultiBody::add_link_force(int link, const Vec3f& f) {
    link_forces_[link] += f;
}

void MultiBody::add_link_torque(int link, const Vec3f& t) {
    link_torques_[link] += t;
}

void MultiBody::clear_forces() {
    for (auto& f : link_forces_) f.setZero();
    for (auto& t : link_torques_) t.setZero();
    for (auto& f : constraint_forces_) f.setZero();
    for (auto& t : constraint_torques_) t.setZero();
    // tau_ is NOT cleared here — joint torques are user-controlled and persist between steps.
    // Users set body.tau before each world.step() call; it is the control input for that step.
}

// ---------------------------------------------------------------------------
// Forward Kinematics
// ---------------------------------------------------------------------------

void MultiBody::forward_kinematics() {
    fk_cache_ = featherstone::forward_kinematics(model_, q_);
    world_transforms_ = fk_cache_.world_transforms;
}

// ---------------------------------------------------------------------------
// Forward Dynamics (replaces 3-pass ABA)
// ---------------------------------------------------------------------------

void MultiBody::compute_accelerations_aba(float dt, bool is_constraint_pass,
                                           const Vec3f& gravity) {
    const int n = num_links();
    const int nv = num_dofs();

    // 1. Update FK cache (needed for force frame conversion and Jacobians)
    fk_cache_ = featherstone::forward_kinematics(model_, q_);
    world_transforms_ = fk_cache_.world_transforms;

    // 2. Build per-link external spatial forces in link-local frame.
    //    featherstone::forward_dynamics() expects f_ext in link-local frame.
    //    MultiBody stores external forces in world frame, so we convert using
    //    rot_world_to_local = world_transforms[i].rotation_matrix().transpose().
    const auto& forces  = is_constraint_pass ? constraint_forces_  : link_forces_;
    const auto& torques = is_constraint_pass ? constraint_torques_ : link_torques_;

    std::vector<SpatialVector> f_ext(n, SpatialVector::Zero());
    for (int i = 0; i < n; ++i) {
        const Vec3f& f_w = forces[i];
        const Vec3f& t_w = torques[i];
        if (f_w.isZero() && t_w.isZero()) continue;

        // Rotate from world frame to link-local frame
        Mat3f Rw = fk_cache_.world_transforms[i].rotation_matrix().transpose();
        Vec3f f_local = Rw * f_w;
        Vec3f t_local = Rw * t_w;

        // Express at link frame origin (I_body is at origin, COM offset accounted for)
        const Vec3f& com = model_.bodies[i].com;
        f_ext[i] = make_spatial(t_local + com.cross(f_local), f_local);
    }

    // 3. Add custom damping to f_ext (if non-zero settings).
    //    Requires link spatial velocities — propagate them with a forward pass.
    if (linear_damping_ != 0.0f || angular_damping_ != 0.0f) {
        std::vector<SpatialVector> sv(n, SpatialVector::Zero());
        for (int i = 0; i < n; ++i) {
            const auto& joint = model_.joints[i];
            const int qdi = dof_offsets_[i];
            const int nv_i = joint.num_qd();

            SpatialVector S_cols[6];
            joint.motion_subspace(S_cols);
            SpatialVector vJ = SpatialVector::Zero();
            for (int k = 0; k < nv_i; ++k) vJ += S_cols[k] * qd_(qdi + k);

            if (joint.parent < 0) {
                sv[i] = vJ;
            } else {
                sv[i] = fk_cache_.parent_transforms[i].apply_motion(sv[joint.parent]) + vJ;
            }

            Vec3f v_omega = spatial_angular(sv[i]);
            Vec3f v_lin   = spatial_linear(sv[i]);

            const float k1l = linear_damping_;
            const float k2l = linear_damping_;
            const float k1a = angular_damping_;
            const float k2a = angular_damping_;
            const float mass_i = model_.bodies[i].mass;
            const Vec3f& inertia_diag = model_.bodies[i].inertia.diagonal();

            Vec3f damp_ang = Vec3f(inertia_diag.x() * v_omega.x(),
                                   inertia_diag.y() * v_omega.y(),
                                   inertia_diag.z() * v_omega.z())
                * (k1a + k2a * v_omega.norm());
            Vec3f damp_lin = mass_i * v_lin * (k1l + k2l * v_lin.norm());

            // Damping forces oppose motion → subtract from bias (add to f_ext as negative)
            f_ext[i] -= make_spatial(damp_ang, damp_lin);
        }
    }

    // 4. Forward dynamics: qdd = H^{-1}(tau - C(q, qd, gravity, f_ext))
    VecXf qdd = featherstone::forward_dynamics(model_, q_, qd_, tau_, gravity, f_ext);

    // 5. Semi-implicit Euler velocity update
    if (!is_constraint_pass && dt > 0.0f) {
        qd_ += qdd * dt;
    }

    // 6. Cache mass matrix + Cholesky factorization for constraint solver
    mass_matrix_cache_ = featherstone::mass_matrix(model_, q_);
    mass_llt_ = mass_matrix_cache_.llt();

    // Store output accelerations
    output_.segment(6, nv) = qdd;
}

// ---------------------------------------------------------------------------
// Constraint Solver Support
// ---------------------------------------------------------------------------

void MultiBody::calc_acceleration_deltas(const float* force_in, float* output) const {
    const int nv = num_dofs();

    // Base DOFs are fixed (world frame does not move)
    std::fill(output, output + 6, 0.0f);

    if (nv == 0) return;

    // Solve H * delta_a = J  (mass matrix Cholesky solve)
    // force_in[6:6+nv] is the joint-space test force (Jacobian row)
    VecXf J = VecXf::Map(force_in + 6, nv);
    VecXf delta_a = mass_llt_.solve(J);
    VecXf::Map(output + 6, nv) = delta_a;
}

void MultiBody::fill_constraint_jacobian(int link, const Vec3f& contact_point,
                                          const Vec3f& normal_ang,
                                          const Vec3f& normal_lin,
                                          float* jac) const {
    const int nv = num_dofs();
    const int n = num_links();

    // Build chain from contact link to root
    std::vector<int> chain;
    int l = link;
    while (l != -1) {
        chain.push_back(l);
        l = model_.joints[l].parent;
    }

    // Initialize all jac entries to zero
    for (int i = 0; i < 6 + nv; ++i) jac[i] = 0.0f;

    // Base DOFs (first 6): always zero — world frame is immovable.
    if (n == 0 || link < 0) return;

    // Contact point relative to world origin
    Vec3f p_minus_com_world = contact_point;

    // Per-link local frames for p and normals.
    // Index 0 = world/base frame (identity rotation), i+1 = link i's frame.
    std::vector<Vec3f> p_local(n + 1);
    std::vector<Vec3f> n_lin_local(n + 1);
    std::vector<Vec3f> n_ang_local(n + 1);

    // World frame (index 0): rot_from_world[0] = Identity
    p_local[0] = p_minus_com_world;
    n_lin_local[0] = normal_lin;
    n_ang_local[0] = normal_ang;

    // Traverse chain from root end to contact link
    for (int a = static_cast<int>(chain.size()) - 1; a >= 0; --a) {
        int i = chain[a];
        const int parent = model_.joints[i].parent;
        const int parent_idx = (parent < 0) ? 0 : (parent + 1);

        // FK cache: parent_transforms[i].E  = rot_from_parent (parent→child rotation)
        //           parent_transforms[i].r  = depends on joint type:
        //             Revolute/Slide/Ball: r = R_joint * t_p2j  (pre-rotated by joint angle)
        //               → R * p - r  =  R * (p - t_p2j)   ✓ position-independent
        //             Free: r = (x,y,z) world position (NOT pre-rotated)
        //               → must use R * (p - r) to correctly transform into body frame
        //               → R * p - r  is WRONG when R ≠ I (off-origin robots drift)
        const Mat3f& R = fk_cache_.parent_transforms[i].E;
        const Vec3f& r = fk_cache_.parent_transforms[i].r;

        n_lin_local[i + 1] = R * n_lin_local[parent_idx];
        n_ang_local[i + 1] = R * n_ang_local[parent_idx];

        // Compute Jacobian entries for this link's joint DOFs
        const auto& joint = model_.joints[i];
        if (joint.type == JointType::Free) {
            p_local[i + 1] = R * (p_local[parent_idx] - r);
        } else {
            p_local[i + 1] = R * p_local[parent_idx] - r;
        }
        const int nv_i = joint.num_qd();
        const int dof_offset = dof_offsets_[i];

        if (nv_i == 0) continue;

        SpatialVector S_cols[6];
        joint.motion_subspace(S_cols);

        switch (joint.type) {
            case JointType::Revolute: {
                Vec3f axis_top = spatial_angular(S_cols[0]);
                Vec3f axis_bottom = spatial_linear(S_cols[0]);
                jac[6 + dof_offset] =
                    n_lin_local[i + 1].dot(axis_top.cross(p_local[i + 1]) + axis_bottom)
                    + n_ang_local[i + 1].dot(axis_top);
                break;
            }
            case JointType::Slide: {
                Vec3f axis_bottom = spatial_linear(S_cols[0]);
                jac[6 + dof_offset] = n_lin_local[i + 1].dot(axis_bottom);
                break;
            }
            case JointType::Ball: {
                for (int d = 0; d < 3; ++d) {
                    Vec3f axis_top = spatial_angular(S_cols[d]);
                    Vec3f axis_bottom = spatial_linear(S_cols[d]);
                    jac[6 + dof_offset + d] =
                        n_lin_local[i + 1].dot(axis_top.cross(p_local[i + 1]) + axis_bottom)
                        + n_ang_local[i + 1].dot(axis_top);
                }
                break;
            }
            case JointType::Free: {
                // 6 DOF: first 3 angular, last 3 linear
                for (int d = 0; d < 3; ++d) {
                    Vec3f axis_top = spatial_angular(S_cols[d]);
                    Vec3f axis_bottom = spatial_linear(S_cols[d]);
                    jac[6 + dof_offset + d] =
                        n_lin_local[i + 1].dot(axis_top.cross(p_local[i + 1]) + axis_bottom)
                        + n_ang_local[i + 1].dot(axis_top);
                }
                for (int d = 3; d < 6; ++d) {
                    Vec3f axis_bottom = spatial_linear(S_cols[d]);
                    jac[6 + dof_offset + d] = n_lin_local[i + 1].dot(axis_bottom);
                }
                break;
            }
            case JointType::Fixed:
                break;
        }
    }
}


// ---------------------------------------------------------------------------
// Position Integration
// ---------------------------------------------------------------------------

void MultiBody::step_positions(float dt) {
    const int n = num_links();

    // Quaternion integration helper (exponential map for large rotations)
    auto quat_integrate = [](const Vec3f& omega, Quatf& quat, bool base_body, float dt) {
        Vec3f angvel;
        if (!base_body) {
            angvel = quat * omega;  // local to global
        } else {
            angvel = omega;
        }

        float fAngle = angvel.norm();
        constexpr float ANGULAR_MOTION_THRESHOLD = 0.5f * 3.14159265f;
        if (fAngle * dt > ANGULAR_MOTION_THRESHOLD) {
            fAngle = 0.5f * 3.14159265f / dt;
        }

        Vec3f axis;
        if (fAngle < 0.001f) {
            axis = angvel * (0.5f * dt - dt * dt * dt * 0.020833333f * fAngle * fAngle);
        } else {
            axis = angvel * (std::sin(0.5f * fAngle * dt) / fAngle);
        }

        float cos_half = std::cos(fAngle * dt * 0.5f);

        if (!base_body) {
            quat = Quatf(cos_half, axis.x(), axis.y(), axis.z()) * quat;
        } else {
            quat = quat * Quatf(cos_half, -axis.x(), -axis.y(), -axis.z());
        }
        quat.normalize();
    };

    for (int i = 0; i < n; ++i) {
        const auto& joint = model_.joints[i];
        const int qi = cfg_offsets_[i];
        const int qdi = dof_offsets_[i];

        switch (joint.type) {
            case JointType::Revolute:
            case JointType::Slide:
                q_(qi) += dt * qd_(qdi);
                break;

            case JointType::Fixed:
                break;

            case JointType::Free: {
                // Position: q[0:3] += dt * qd[3:6] (linear velocity)
                q_(qi + 0) += dt * qd_(qdi + 3);
                q_(qi + 1) += dt * qd_(qdi + 4);
                q_(qi + 2) += dt * qd_(qdi + 5);

                // Quaternion: q[3:7] = [qx,qy,qz,qw]
                // Angular velocity: qd[0:3]
                Vec3f omega(qd_(qdi), qd_(qdi + 1), qd_(qdi + 2));
                Quatf quat(q_(qi + 6), q_(qi + 3), q_(qi + 4), q_(qi + 5)); // Eigen: w,x,y,z
                quat_integrate(omega, quat, true, dt);
                q_(qi + 3) = quat.x();
                q_(qi + 4) = quat.y();
                q_(qi + 5) = quat.z();
                q_(qi + 6) = quat.w();
                break;
            }

            case JointType::Ball: {
                // q[0:4] = [qx,qy,qz,qw], qd[0:3] = angular velocity
                Vec3f omega(qd_(qdi), qd_(qdi + 1), qd_(qdi + 2));
                Quatf quat(q_(qi + 3), q_(qi), q_(qi + 1), q_(qi + 2)); // Eigen: w,x,y,z
                quat_integrate(omega, quat, false, dt);
                q_(qi + 0) = quat.x();
                q_(qi + 1) = quat.y();
                q_(qi + 2) = quat.z();
                q_(qi + 3) = quat.w();
                break;
            }
        }
    }
}

}  // namespace novaphy
