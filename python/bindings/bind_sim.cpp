#include <memory>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "novaphy/core/control.h"
#include "novaphy/core/device.h"
#include "novaphy/core/model.h"
#include "novaphy/core/model_builder.h"
#include "novaphy/dynamics/solver_base.h"
#include "novaphy/dynamics/solver_sequential_impulse.h"
#include "novaphy/dynamics/xpbd_solver.h"
#include "novaphy/fluid/pbf_solver.h"
#include "novaphy/sim/state.h"
#include "novaphy/sim/world.h"

namespace py = pybind11;
using namespace novaphy;

void bind_sim(py::module_& m) {
    // --- DeviceType enum ---
    py::enum_<DeviceType>(m, "DeviceType", R"pbdoc(
        Compute backend type for simulation.
    )pbdoc")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("Vulkan", DeviceType::Vulkan)
        .value("Metal", DeviceType::Metal);

    // --- Device ---
    py::class_<Device>(m, "Device", R"pbdoc(
        Lightweight device descriptor for compute backend selection.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("type", &Device::type)
        .def_readwrite("ordinal", &Device::ordinal)
        .def_static("cpu", &Device::cpu)
        .def_static("cuda", &Device::cuda, py::arg("ordinal") = 0)
        .def("__eq__", &Device::operator==)
        .def("__ne__", &Device::operator!=);

    // --- JointTargetMode enum ---
    py::enum_<JointTargetMode>(m, "JointTargetMode", R"pbdoc(
        Joint drive target mode.
    )pbdoc")
        .value("Off", JointTargetMode::Off)
        .value("TargetPosition", JointTargetMode::TargetPosition)
        .value("TargetVelocity", JointTargetMode::TargetVelocity);

    // --- JointDrive ---
    py::class_<JointDrive>(m, "JointDrive", R"pbdoc(
        Per-joint PD drive configuration.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("mode", &JointDrive::mode)
        .def_readwrite("target_position", &JointDrive::target_position)
        .def_readwrite("target_velocity", &JointDrive::target_velocity)
        .def_readwrite("stiffness", &JointDrive::stiffness)
        .def_readwrite("damping", &JointDrive::damping)
        .def_readwrite("force_limit", &JointDrive::force_limit);

    // --- Control ---
    py::class_<Control>(m, "Control", R"pbdoc(
        Unified runtime control input for one simulation step.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("joint_forces", &Control::joint_forces)
        .def_readwrite("articulation_joint_forces", &Control::articulation_joint_forces)
        .def_readwrite("joint_drives", &Control::joint_drives)
        .def_readwrite("body_forces", &Control::body_forces)
        .def_readwrite("body_torques", &Control::body_torques);


    py::class_<CollisionFilterPair>(m, "CollisionFilterPair", R"pbdoc(
        Disabled collision pair stored in shape-index space.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("shape_a", &CollisionFilterPair::shape_a)
        .def_readwrite("shape_b", &CollisionFilterPair::shape_b);
    // --- ModelBuilder ---
    py::class_<ModelBuilder>(m, "ModelBuilder", R"pbdoc(
        Builder for constructing an immutable simulation model.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Creates an empty model builder.
        )pbdoc")
        .def(
            "add_body",
            [](ModelBuilder& self, const RigidBody& body, py::object transform) {
                Transform t =
                    transform.is_none() ? Transform::identity() : transform.cast<Transform>();
                return self.add_body(body, t);
            },
            py::arg("body"),
            py::arg("transform") = py::none(),
            R"pbdoc(
                Adds a rigid body and returns its index.

                Args:
                    body (RigidBody): Body mass and inertia properties.
                    transform (Transform): Initial world transform.

                Returns:
                    int: New body index.
            )pbdoc")
        .def("add_shape", &ModelBuilder::add_shape,
             py::arg("shape"),
             R"pbdoc(
                 Adds a collision shape and returns its index.

                 Args:
                     shape (CollisionShape): Shape attached to a body or world.

                 Returns:
                     int: New shape index.
             )pbdoc")
        .def("add_ground_plane", &ModelBuilder::add_ground_plane,
             py::arg("y") = 0.0f,
             py::arg("friction") = 0.5f,
             py::arg("restitution") = 0.0f,
             R"pbdoc(
                 Adds an infinite ground plane shape.

                 Args:
                     y (float): Plane offset along +Y world axis in meters.
                     friction (float): Friction coefficient.
                     restitution (float): Restitution coefficient.

                 Returns:
                     int: New shape index.
             )pbdoc")
        .def(
            "add_shape_box",
            [](ModelBuilder& self, const Vec3f& half_extents, py::object transform, float density,
               float friction, float restitution, bool is_static) {
                Transform t =
                    transform.is_none() ? Transform::identity() : transform.cast<Transform>();
                return self.add_shape_box(half_extents, t, density, friction, restitution,
                                           is_static);
            },
            py::arg("half_extents"),
            py::arg("transform") = py::none(),
            py::arg("density") = 1000.0f,
            py::arg("friction") = 0.5f,
            py::arg("restitution") = 0.3f,
            py::arg("is_static") = false,
            R"pbdoc(
                Adds a box body+shape in one call with density-based mass.

                Args:
                    half_extents (Vector3): Box half extents in meters.
                    transform (Transform): Initial world transform.
                    density (float): Mass density in kg/m^3.
                    friction (float): Friction coefficient.
                    restitution (float): Restitution coefficient.
                    is_static (bool): If True, body is immovable.

                Returns:
                    int: New body index.
            )pbdoc")
        .def(
            "add_shape_sphere",
            [](ModelBuilder& self, float radius, py::object transform, float density, float friction,
               float restitution, bool is_static) {
                Transform t =
                    transform.is_none() ? Transform::identity() : transform.cast<Transform>();
                return self.add_shape_sphere(radius, t, density, friction, restitution, is_static);
            },
            py::arg("radius"),
            py::arg("transform") = py::none(),
            py::arg("density") = 1000.0f,
            py::arg("friction") = 0.5f,
            py::arg("restitution") = 0.3f,
            py::arg("is_static") = false,
            R"pbdoc(
                Adds a sphere body+shape in one call with density-based mass.

                Args:
                    radius (float): Sphere radius in meters.
                    transform (Transform): Initial world transform.
                    density (float): Mass density in kg/m^3.
                    friction (float): Friction coefficient.
                    restitution (float): Restitution coefficient.
                    is_static (bool): If True, body is immovable.

                Returns:
                    int: New body index.
            )pbdoc")
        .def(
            "add_shape_cylinder",
            [](ModelBuilder& self, float radius, float half_length, py::object transform,
               float density, float friction, float restitution, bool is_static) {
                Transform t =
                    transform.is_none() ? Transform::identity() : transform.cast<Transform>();
                return self.add_shape_cylinder(radius, half_length, t, density, friction,
                                                 restitution, is_static);
            },
            py::arg("radius"), py::arg("half_length"),
            py::arg("transform") = py::none(),
            py::arg("density") = 1000.0f,
            py::arg("friction") = 0.5f,
            py::arg("restitution") = 0.3f,
            py::arg("is_static") = false,
            R"pbdoc(
                Adds a cylinder body+shape in one call with density-based mass.

                Args:
                    radius (float): Cylinder radius in meters.
                    half_length (float): Half-length along local Z axis in meters.
                    transform (Transform): Initial world transform.
                    density (float): Mass density in kg/m^3.
                    friction (float): Friction coefficient.
                    restitution (float): Restitution coefficient.
                    is_static (bool): If True, body is immovable.

                Returns:
                    int: New body index.
            )pbdoc")
        .def("add_shape_to_body", &ModelBuilder::add_shape_to_body,
             py::arg("body_index"), py::arg("shape"),
             R"pbdoc(
                 Adds a collision shape attached to an existing body.

                 Args:
                     body_index (int): Existing body index.
                     shape (CollisionShape): Shape descriptor.

                 Returns:
                     int: New shape index.
             )pbdoc")
        .def("add_collision_filter", &ModelBuilder::add_collision_filter,
             py::arg("shape_a"), py::arg("shape_b"),
             R"pbdoc(
                 Disables collision between two shapes.

                 Args:
                     shape_a (int): First shape index.
                     shape_b (int): Second shape index.
             )pbdoc")
        .def("set_gravity", &ModelBuilder::set_gravity,
             py::arg("gravity"),
             R"pbdoc(
                 Sets gravity vector for the built model.

                 Args:
                     gravity (Vector3): Gravity in world coordinates (m/s^2).
             )pbdoc")
        .def("build", &ModelBuilder::build, R"pbdoc(
            Builds an immutable `Model` from accumulated bodies and shapes.

            Returns:
                Model: Immutable model object.
        )pbdoc")
        .def_property_readonly("num_bodies", &ModelBuilder::num_bodies, R"pbdoc(
            int: Number of currently added bodies.
        )pbdoc")
        .def_property_readonly("num_shapes", &ModelBuilder::num_shapes, R"pbdoc(
            int: Number of currently added shapes.
        )pbdoc");

    // --- Model ---
    py::class_<Model>(m, "Model", R"pbdoc(
        Immutable collection of rigid bodies and collision shapes.
    )pbdoc")
        .def_property_readonly("num_bodies", &Model::num_bodies, R"pbdoc(
            int: Number of bodies in the model.
        )pbdoc")
        .def_property_readonly("num_shapes", &Model::num_shapes, R"pbdoc(
            int: Number of shapes in the model.
        )pbdoc")
        .def_readonly("bodies", &Model::bodies, R"pbdoc(
            list[RigidBody]: Body inertial properties.
        )pbdoc")
        .def_readonly("initial_transforms", &Model::initial_transforms, R"pbdoc(
            list[Transform]: Initial world transform per body.
        )pbdoc")
        .def_readonly("collision_filter_pairs", &Model::collision_filter_pairs, R"pbdoc(
            list[CollisionFilterPair]: Disabled shape pairs used to skip narrowphase.
        )pbdoc")
        .def_readwrite("shapes", &Model::shapes, R"pbdoc(
            list[CollisionShape]: Collision shape definitions.
        )pbdoc")
        .def_readwrite("gravity", &Model::gravity, R"pbdoc(
            Vector3: World gravity vector (m/s^2). Default is (0, -9.81, 0).
        )pbdoc")
        .def_readwrite("fluid_blocks", &Model::fluid_blocks, R"pbdoc(
            list[FluidBlockDef]: Fluid particle block definitions.
        )pbdoc")
        .def_readwrite("fluid_materials", &Model::fluid_materials, R"pbdoc(
            list[FluidMaterial]: Optional multi-material table (indexed by block.material_index).
        )pbdoc")
        .def_readwrite("articulations", &Model::articulations, R"pbdoc(
            list[Articulation]: Tree-structured articulated robots.
        )pbdoc");

    // --- SolverSettings ---
    py::class_<SolverSettings>(m, "SolverSettings", R"pbdoc(
        Configuration for the contact solver iteration and stabilization.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Creates solver settings with default stable values.
        )pbdoc")
        .def_readwrite("velocity_iterations", &SolverSettings::velocity_iterations, R"pbdoc(
            int: Number of PGS velocity iterations per time step.
        )pbdoc")
        .def_readwrite("baumgarte", &SolverSettings::baumgarte, R"pbdoc(
            float: Position-error correction factor (dimensionless).
        )pbdoc")
        .def_readwrite("slop", &SolverSettings::slop, R"pbdoc(
            float: Penetration allowance before correction (meters).
        )pbdoc")
        .def_readwrite("warm_starting", &SolverSettings::warm_starting, R"pbdoc(
            bool: Reuse previous-frame impulses for faster convergence.
        )pbdoc")
        .def_readwrite("sleep_enabled", &SolverSettings::sleep_enabled, R"pbdoc(
            bool: Enable sleep mechanism to freeze stationary bodies.
        )pbdoc")
        .def_readwrite("sleep_energy_threshold", &SolverSettings::sleep_energy_threshold, R"pbdoc(
            float: Kinetic energy threshold for sleep (universal, all scenarios).
        )pbdoc")
        .def_readwrite("sleep_time_required", &SolverSettings::sleep_time_required, R"pbdoc(
            float: Time below threshold before sleeping (seconds).
        )pbdoc")
        .def_readwrite("sleep_ema_alpha", &SolverSettings::sleep_ema_alpha, R"pbdoc(
            float: EMA smoothing factor for energy (0-1).
        )pbdoc");

    // --- SolverBase (abstract) ---
    py::class_<SolverBase>(m, "SolverBase", R"pbdoc(
        Abstract base class for all physics solvers.
    )pbdoc")
        .def("name", &SolverBase::name)
        .def_property_readonly("contacts",
             &SolverBase::contacts,
             py::return_value_policy::reference_internal);

    // --- SolverSequentialImpulse ---
    py::class_<SolverSequentialImpulse, SolverBase>(m, "SolverSequentialImpulse", R"pbdoc(
        PGS contact solver for free rigid bodies.
    )pbdoc")
        .def(py::init<SolverSettings>(),
             py::arg("settings") = SolverSettings{});


    // --- SimState ---
    py::class_<SimState>(m, "SimState", R"pbdoc(
        Mutable simulation state for all bodies in the world.

        Can be created independently via ``SimState.from_model()`` and
        deep-copied with ``clone()`` for parallel rollout / checkpoint.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Creates an empty simulation state.
        )pbdoc")
        .def_static("from_model",
             [](const Model& model, std::optional<Device> device) {
                 return SimState::from_model(model, device.value_or(Device::cpu()));
             },
             py::arg("model"),
             py::arg("device") = py::none(),
             R"pbdoc(
                 Creates a SimState initialized from a model's initial transforms.

                 Args:
                     model (Model): Model providing body count and initial transforms.
                     device (Device): Target compute device (default: CPU).

                 Returns:
                     SimState: Newly constructed state.
             )pbdoc")
        .def("clone", &SimState::clone, R"pbdoc(
            Deep-copies this state into an independent SimState.

            Returns:
                SimState: Independent copy.
        )pbdoc")
        .def_property_readonly("device", &SimState::device, R"pbdoc(
            Device: Compute device associated with this state.
        )pbdoc")
        .def_readonly("transforms", &SimState::transforms, R"pbdoc(
            list[Transform]: World transforms per body.
        )pbdoc")
        .def_readonly("linear_velocities", &SimState::linear_velocities, R"pbdoc(
            list[Vector3]: Linear velocities in world coordinates (m/s).
        )pbdoc")
        .def_readonly("angular_velocities", &SimState::angular_velocities, R"pbdoc(
            list[Vector3]: Angular velocities in world coordinates (rad/s).
        )pbdoc")
        .def("set_linear_velocity", &SimState::set_linear_velocity,
             py::arg("body_index"), py::arg("velocity"),
             R"pbdoc(
                 Sets one body's linear velocity.

                 Args:
                     body_index (int): Body index.
                     velocity (Vector3): Velocity in world coordinates (m/s).

                 Returns:
                     None
             )pbdoc")
        .def("set_angular_velocity", &SimState::set_angular_velocity,
             py::arg("body_index"), py::arg("velocity"),
             R"pbdoc(
                 Sets one body's angular velocity.

                 Args:
                     body_index (int): Body index.
                     velocity (Vector3): Angular velocity in world coordinates (rad/s).

                 Returns:
                     None
             )pbdoc")
        .def("apply_force", &SimState::apply_force, py::arg("body_index"), py::arg("force"),
             R"pbdoc(
                 Accumulates an external force at body center of mass.

                 Args:
                     body_index (int): Body index.
                     force (Vector3): Force in world coordinates (N).

                 Returns:
                     None
             )pbdoc")
        .def("apply_torque", &SimState::apply_torque, py::arg("body_index"), py::arg("torque"),
             R"pbdoc(
                 Accumulates an external torque on a body.

                 Args:
                     body_index (int): Body index.
                     torque (Vector3): Torque in world coordinates (N*m).

                 Returns:
                     None
             )pbdoc")
        .def("get_transforms_numpy",
             [](const SimState& s) -> py::tuple {
                 const int N = static_cast<int>(s.transforms.size());
                 // Allocate (N,3) positions and (N,4) quaternions xyzw
                 py::array_t<float> pos({N, 3});
                 py::array_t<float> quat({N, 4});
                 auto p = pos.mutable_unchecked<2>();
                 auto q = quat.mutable_unchecked<2>();
                 for (int i = 0; i < N; ++i) {
                     const auto& t = s.transforms[i];
                     p(i, 0) = t.position.x();
                     p(i, 1) = t.position.y();
                     p(i, 2) = t.position.z();
                     q(i, 0) = t.rotation.x();
                     q(i, 1) = t.rotation.y();
                     q(i, 2) = t.rotation.z();
                     q(i, 3) = t.rotation.w();
                 }
                 return py::make_tuple(pos, quat);
             },
             R"pbdoc(
                 Returns all body transforms as a pair of NumPy arrays in one
                 C++ call, avoiding per-body Python鈫擟++ boundary crossings.

                 Returns:
                     tuple[np.ndarray, np.ndarray]:
                         positions ``(N, 3)`` float32 and
                         quaternions ``(N, 4)`` float32 in ``[x, y, z, w]`` order.
             )pbdoc")
        .def_property("fluid_state",
             [](SimState& s) -> ParticleState& { return s.fluid_state; },
             [](SimState& s, const ParticleState& p) { s.fluid_state = p; },
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 ParticleState: Fluid particle state (positions, velocities, etc.).
             )pbdoc")
        .def_readonly("q", &SimState::q, R"pbdoc(
            list[ndarray]: Generalized position vectors per articulation.
        )pbdoc")
        .def_readonly("qd", &SimState::qd, R"pbdoc(
            list[ndarray]: Generalized velocity vectors per articulation.
        )pbdoc")
        .def("set_q",
             [](SimState& s, int art_idx, const VecXf& q) { s.q.at(art_idx) = q; },
             py::arg("art_idx"), py::arg("q"),
             R"pbdoc(
                 Sets the generalized positions for one articulation.

                 Args:
                     art_idx (int): Articulation index.
                     q (ndarray): New generalized positions.
             )pbdoc")
        .def("set_qd",
             [](SimState& s, int art_idx, const VecXf& qd) { s.qd.at(art_idx) = qd; },
             py::arg("art_idx"), py::arg("qd"),
             R"pbdoc(
                 Sets the generalized velocities for one articulation.

                 Args:
                     art_idx (int): Articulation index.
                     qd (ndarray): New generalized velocities.
             )pbdoc")
        .def("get_transforms_into",
             [](const SimState& s,
                py::array_t<float, py::array::c_style | py::array::forcecast> pos_arr,
                py::array_t<float, py::array::c_style | py::array::forcecast> quat_arr) {
                 const int N = static_cast<int>(s.transforms.size());
                 auto p = pos_arr.mutable_unchecked<2>();
                 auto q = quat_arr.mutable_unchecked<2>();
                 for (int i = 0; i < N; ++i) {
                     const auto& t = s.transforms[i];
                     p(i,0)=t.position.x(); p(i,1)=t.position.y(); p(i,2)=t.position.z();
                     q(i,0)=t.rotation.x(); q(i,1)=t.rotation.y();
                     q(i,2)=t.rotation.z(); q(i,3)=t.rotation.w();
                 }
             },
             py::arg("out_positions"), py::arg("out_quats"),
             R"pbdoc(Fills pre-allocated (N,3) and (N,4) float32 arrays in-place.)pbdoc");

    // --- Module-level batch transform ---
    m.def("batch_transform_vertices",
        [](py::array_t<float,   py::array::c_style | py::array::forcecast> positions,
           py::array_t<float,   py::array::c_style | py::array::forcecast> quats,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> body_indices,
           py::array_t<float,   py::array::c_style | py::array::forcecast> local_verts,
           py::array_t<float,   py::array::c_style | py::array::forcecast> out_verts) {
             const int N = static_cast<int>(body_indices.shape(0));
             const int V = static_cast<int>(local_verts.shape(1));
             auto p   = positions.unchecked<2>();
             auto q   = quats.unchecked<2>();
             auto idx = body_indices.unchecked<1>();
             auto lv  = local_verts.unchecked<3>();
             auto out = out_verts.mutable_unchecked<2>();
             for (int i = 0; i < N; ++i) {
                 const int bi = idx(i);
                 const float qx=q(bi,0), qy=q(bi,1), qz=q(bi,2), qw=q(bi,3);
                 const float R00=1-2*(qy*qy+qz*qz), R01=2*(qx*qy-qz*qw), R02=2*(qx*qz+qy*qw);
                 const float R10=2*(qx*qy+qz*qw),   R11=1-2*(qx*qx+qz*qz), R12=2*(qy*qz-qx*qw);
                 const float R20=2*(qx*qz-qy*qw),   R21=2*(qy*qz+qx*qw),   R22=1-2*(qx*qx+qy*qy);
                 const float px=p(bi,0), py_=p(bi,1), pz=p(bi,2);
                 const int base = i * V;
                 for (int j = 0; j < V; ++j) {
                     const float vx=lv(i,j,0), vy=lv(i,j,1), vz=lv(i,j,2);
                     out(base+j,0) = R00*vx + R01*vy + R02*vz + px;
                     out(base+j,1) = R10*vx + R11*vy + R12*vz + py_;
                     out(base+j,2) = R20*vx + R21*vy + R22*vz + pz;
                 }
             }
        },
        py::arg("positions"), py::arg("quats"),
        py::arg("body_indices"), py::arg("local_verts"), py::arg("out_vertices"),
        py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
            Fused batched vertex transform from snapshot pos/quat arrays.

            For each body i, reads position and quaternion by body_indices[i], builds the
            3x3 rotation matrix inline, applies it to local_verts[i], and writes world
            vertices into out_vertices[i*V .. (i+1)*V].  Single C++ pass: no intermediate
            rotation-matrix array, no Python overhead, GIL released.

            Args:
                positions (np.ndarray): (N_bodies, 3) float32 world positions (snapshot).
                quats (np.ndarray): (N_bodies, 4) float32 quaternions [x,y,z,w] (snapshot).
                body_indices (np.ndarray): (N,) int32 indices into positions/quats.
                local_verts (np.ndarray): (N, V, 3) float32 local-space vertices.
                out_vertices (np.ndarray): (N*V, 3) float32 pre-allocated output buffer.
        )pbdoc");

    // --- World ---
    py::class_<World>(m, "World", R"pbdoc(
        Top-level simulation container with pluggable solver.
    )pbdoc")
        .def(
            py::init([](const Model& model, py::object solver_settings, py::object xpbd_settings,
                        py::object pbf_settings, float fluid_boundary_extent) {
                SolverSettings ss =
                    solver_settings.is_none() ? SolverSettings{} : solver_settings.cast<SolverSettings>();
                XPBDSolverSettings xs =
                    xpbd_settings.is_none() ? XPBDSolverSettings{} : xpbd_settings.cast<XPBDSolverSettings>();
                PBFSettings ps =
                    pbf_settings.is_none() ? PBFSettings{} : pbf_settings.cast<PBFSettings>();
                return std::make_unique<World>(model, ss, xs, ps, fluid_boundary_extent);
            }),
             py::arg("model"),
             py::arg("solver_settings") = py::none(),
             py::arg("xpbd_settings") = py::none(),
             py::arg("pbf_settings") = py::none(),
             py::arg("fluid_boundary_extent") = 1.0f,
             R"pbdoc(
                 Creates a unified simulation world from an immutable model.

                 Args:
                     model (Model): Immutable model definition.
                     solver_settings (SolverSettings): Contact solver parameters for rigid bodies.
                     xpbd_settings (XPBDSolverSettings): XPBD solver parameters for articulations.
                     pbf_settings (PBFSettings): PBF solver parameters for fluid particles.
                     fluid_boundary_extent (float): Half-extent for plane boundary sampling (m).
             )pbdoc")
        .def("step", static_cast<void (World::*)(float)>(&World::step),
             py::arg("dt"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Advances simulation by one fixed time step using internal state.

                 Args:
                     dt (float): Time step in seconds.
             )pbdoc")
        .def("step_with_control",
             static_cast<void (World::*)(SimState&, const Control&, float)>(&World::step),
             py::arg("state"), py::arg("control"), py::arg("dt"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Advances simulation with explicit state and control inputs.

                 Args:
                     state (SimState): Mutable simulation state.
                     control (Control): External control inputs.
                     dt (float): Time step in seconds.
             )pbdoc")
        .def("set_gravity", &World::set_gravity, py::arg("gravity"),
             R"pbdoc(
                 Sets the world gravity vector.

                 Args:
                     gravity (Vector3): Gravity in world coordinates (m/s^2).
             )pbdoc")
        .def_property_readonly("gravity", &World::gravity, R"pbdoc(
            Vector3: Current gravity vector in world coordinates (m/s^2).
        )pbdoc")
        .def_property_readonly("state", py::overload_cast<>(&World::state),
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 SimState: Mutable world state (reference).
             )pbdoc")
        .def_property_readonly("model", &World::model,
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 Model: Immutable world model (reference).
             )pbdoc")
        .def_property_readonly("contacts", &World::contacts,
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 list[ContactPoint]: Contact points generated during last step.
             )pbdoc")
        .def_property_readonly("solver_settings",
             &World::solver_settings,
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 SolverSettings: Contact solver settings (for default PGS solver).
             )pbdoc")
        .def_property_readonly("performance_monitor",
             py::overload_cast<>(&World::performance_monitor),
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 PerformanceMonitor: Runtime performance monitor for this world.
             )pbdoc")
        .def("apply_force", &World::apply_force,
             py::arg("body_index"), py::arg("force"),
             R"pbdoc(
                 Applies an external force for the next simulation step.

                 Args:
                     body_index (int): Body index.
                     force (Vector3): Force in world coordinates (N).
             )pbdoc")
        .def("apply_torque", &World::apply_torque,
             py::arg("body_index"), py::arg("torque"),
             R"pbdoc(
                 Applies an external torque for the next simulation step.

                 Args:
                     body_index (int): Body index.
                     torque (Vector3): Torque in world coordinates (N*m).
             )pbdoc")
        .def_property_readonly("xpbd_solver",
             py::overload_cast<>(&World::xpbd_solver),
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 XPBDSolver: Articulated XPBD solver (reference).
             )pbdoc")
        .def_property_readonly("boundary_particles",
             &World::boundary_particles,
             py::return_value_policy::reference_internal,
             R"pbdoc(
                 list[BoundaryParticle]: Boundary particles sampled from rigid shapes.
             )pbdoc");
}




