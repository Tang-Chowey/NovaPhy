#include "novaphy/vbd/vbd_world.h"

#include "novaphy/vbd/vbd_solver.h"

namespace novaphy {

struct ImplCPU : VBDWorld::Impl {
    Model model_;
    VBDConfig config_;
    SimState state_;
    VbdSolver solver;

    ImplCPU(const Model& m, const VBDConfig& cfg)
        : model_(m), config_(cfg), solver(cfg) {
        state_.init(model_.initial_transforms);
        solver.set_model(model_);
    }

    void step_one() override { solver.step(model_, state_); }
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

static std::unique_ptr<VBDWorld::Impl> create_cpu_impl(const Model& model, const VBDConfig& config) {
    return std::make_unique<ImplCPU>(model, config);
}

std::unique_ptr<VBDWorld::Impl> create_vbd_world_impl(const Model& model, const VBDConfig& config) {
#if defined(NOVAPHY_VBD_CUDA)
    if (config.backend == VbdBackend::CUDA)
        return create_cuda_impl(model, config);
#endif
    return create_cpu_impl(model, config);
}

VBDWorld::VBDWorld(const Model& model, const VBDConfig& config)
    : impl_(create_vbd_world_impl(model, config)) {}

VBDWorld::~VBDWorld() = default;

VBDWorld::VBDWorld(VBDWorld&&) noexcept = default;
VBDWorld& VBDWorld::operator=(VBDWorld&&) noexcept = default;

void VBDWorld::step() {
    impl_->step_one();
}

void VBDWorld::clear_forces() { impl_->clear_forces(); }

void VBDWorld::add_ignore_collision(int body_a, int body_b) {
    impl_->add_ignore_collision(body_a, body_b);
}

int VBDWorld::add_joint(int body_a, int body_b,
                        const Vec3f& rA, const Vec3f& rB,
                        float stiffnessLin, float stiffnessAng, float fracture) {
    return impl_->add_joint(body_a, body_b, rA, rB, stiffnessLin, stiffnessAng, fracture);
}

int VBDWorld::add_spring(int body_a, int body_b,
                         const Vec3f& rA, const Vec3f& rB,
                         float stiffness, float rest) {
    return impl_->add_spring(body_a, body_b, rA, rB, stiffness, rest);
}

SimState& VBDWorld::state() { return impl_->state(); }
const SimState& VBDWorld::state() const { return impl_->state(); }

const Model& VBDWorld::model() const { return impl_->model(); }

const VBDConfig& VBDWorld::config() const { return impl_->config(); }

}  // namespace novaphy
