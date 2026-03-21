/**
 * @file vbd_world_cuda.cpp
 * @brief CUDA backend for VBDWorld: ImplCUDA::step_one() calls only step_cuda(), no CPU solver code.
 * Host-only; no device code. Compiled as .cpp when NOVAPHY_WITH_VBD_CUDA=ON.
 */
#include "novaphy/vbd/vbd_world.h"
#include "novaphy/vbd/vbd_solver.h"

namespace novaphy {

struct ImplCUDA : VBDWorld::Impl {
    Model model_;
    VBDConfig config_;
    SimState state_;
    VbdSolver solver;

    ImplCUDA(const Model& m, const VBDConfig& cfg)
        : model_(m), config_(cfg), solver(cfg) {
        state_.init(model_.initial_transforms);
        solver.set_model(model_);
    }

    void step_one() override { solver.step_cuda(model_, state_); }
    void clear_forces() override { solver.clear_forces(); }
    void add_ignore_collision(int a, int b) override { solver.add_ignore_collision(a, b); }
    int add_joint(int a, int b, const Vec3f& rA, const Vec3f& rB,
                  float kLin, float kAng, float fracture) override {
        return solver.add_joint(a, b, rA, rB, kLin, kAng, fracture);
    }
    int add_spring(int a, int b, const Vec3f& rA, const Vec3f& rB,
                   float stiffness, float rest) override {
        return solver.add_spring(a, b, rA, rB, stiffness, rest);
    }
    SimState& state() override { return state_; }
    const SimState& state() const override { return state_; }
    const Model& model() const override { return model_; }
    const VBDConfig& config() const override { return config_; }
};

std::unique_ptr<VBDWorld::Impl> create_cuda_impl(const Model& model, const VBDConfig& config) {
    return std::make_unique<ImplCUDA>(model, config);
}

}  // namespace novaphy
