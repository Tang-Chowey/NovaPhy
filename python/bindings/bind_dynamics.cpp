#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "novaphy/core/articulation.h"
#include "novaphy/core/joint.h"
#include "novaphy/core/control.h"
#include "novaphy/dynamics/articulated_solver.h"
#include "novaphy/dynamics/xpbd_solver.h"
#include "novaphy/dynamics/featherstone.h"

namespace py = pybind11;
using namespace novaphy;

void bind_dynamics(py::module_& m) {
    // --- JointType ---
    py::enum_<JointType>(m, "JointType", R"pbdoc(
        Joint model type used by articulated-body dynamics.
    )pbdoc")
        .value("Revolute", JointType::Revolute, R"pbdoc(
            One rotational degree of freedom around `axis` (radians).
        )pbdoc")
        .value("Fixed", JointType::Fixed, R"pbdoc(
            Zero degree-of-freedom rigid attachment.
        )pbdoc")
        .value("Free", JointType::Free, R"pbdoc(
            Six degree-of-freedom floating base joint.
        )pbdoc")
        .value("Slide", JointType::Slide, R"pbdoc(
            One translational degree of freedom along `axis` (meters).
        )pbdoc")
        .value("Ball", JointType::Ball, R"pbdoc(
            Three rotational degrees of freedom for orientation.
        )pbdoc");

    // --- Joint ---
    py::class_<Joint>(m, "Joint", R"pbdoc(
        Joint metadata and kinematic parameters for one articulation link.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Creates a default joint descriptor.
        )pbdoc")
        .def_readwrite("type", &Joint::type, R"pbdoc(
            Joint type enumeration controlling motion subspace.
        )pbdoc")
        .def_readwrite("axis", &Joint::axis, R"pbdoc(
            Joint axis vector used by revolute/slide joints in local coordinates.
        )pbdoc")
        .def_readwrite("parent", &Joint::parent, R"pbdoc(
            Parent link index (`-1` denotes root/world parent).
        )pbdoc")
        .def_readwrite("parent_to_joint", &Joint::parent_to_joint, R"pbdoc(
            Transform from parent-link frame to joint frame.
        )pbdoc")
        .def_readwrite("limit_enabled", &Joint::limit_enabled, R"pbdoc(
            bool: Whether lower/upper position limits are active.
        )pbdoc")
        .def_readwrite("lower_limit", &Joint::lower_limit, R"pbdoc(
            float: Lower position limit for revolute/slide joints.
        )pbdoc")
        .def_readwrite("upper_limit", &Joint::upper_limit, R"pbdoc(
            float: Upper position limit for revolute/slide joints.
        )pbdoc")
        .def_readwrite("armature", &Joint::armature, R"pbdoc(
            float: Rotor inertia reflected at joint (kg*m^2).
        )pbdoc")
        .def_readwrite("damping", &Joint::damping, R"pbdoc(
            float: Viscous damping coefficient (N*m*s/rad).
        )pbdoc")
        .def_readwrite("friction", &Joint::friction, R"pbdoc(
            float: Joint friction torque/force (N*m or N).
        )pbdoc")
        .def_readwrite("effort_limit", &Joint::effort_limit, R"pbdoc(
            float: Maximum joint torque/force magnitude.
        )pbdoc")
        .def_readwrite("velocity_limit", &Joint::velocity_limit, R"pbdoc(
            float: Maximum joint velocity magnitude.
        )pbdoc")
        .def("num_q", &Joint::num_q, R"pbdoc(
            Returns the number of generalized position coordinates for this joint.

            Returns:
                int: Number of position coordinates.
        )pbdoc")
        .def("num_qd", &Joint::num_qd, R"pbdoc(
            Returns the number of generalized velocity coordinates for this joint.

            Returns:
                int: Number of velocity coordinates.
        )pbdoc");

    // --- Articulation ---
    py::class_<Articulation>(m, "Articulation", R"pbdoc(
        Tree-structured multibody model used by Featherstone dynamics routines.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Creates an empty articulation model.
        )pbdoc")
        .def_readwrite("joints", &Articulation::joints, R"pbdoc(
            Ordered joint list (one joint per link).
        )pbdoc")
        .def_readwrite("bodies", &Articulation::bodies, R"pbdoc(
            Per-link rigid-body inertial properties.
        )pbdoc")
        .def("num_links", &Articulation::num_links, R"pbdoc(
            Returns the number of links in the articulation.

            Returns:
                int: Number of links.
        )pbdoc")
        .def("total_q", &Articulation::total_q, R"pbdoc(
            Returns the total generalized-position dimension.

            Returns:
                int: Size of the full `q` vector.
        )pbdoc")
        .def("total_qd", &Articulation::total_qd, R"pbdoc(
            Returns the total generalized-velocity dimension.

            Returns:
                int: Size of the full `qd` vector.
        )pbdoc")
        .def("q_start", &Articulation::q_start, py::arg("link"), R"pbdoc(
            Returns the starting index of a link's position block in `q`.

            Args:
                link (int): Link index in the articulation.

            Returns:
                int: Start index into the generalized-position vector.
        )pbdoc")
        .def("qd_start", &Articulation::qd_start, py::arg("link"), R"pbdoc(
            Returns the starting index of a link's velocity block in `qd`.

            Args:
                link (int): Link index in the articulation.

            Returns:
                int: Start index into the generalized-velocity vector.
        )pbdoc")
        .def("build_spatial_inertias", &Articulation::build_spatial_inertias, R"pbdoc(
            Builds per-link spatial inertia matrices from rigid-body properties.

            Returns:
                None
        )pbdoc");

    // --- Featherstone algorithms ---
    m.def("forward_kinematics", [](const Articulation& model, const VecXf& q) {
        return featherstone::forward_kinematics(model, q).world_transforms;
    }, py::arg("model"), py::arg("q"),
    R"pbdoc(
        Computes world-frame transforms for all links from generalized positions.

        Args:
            model (Articulation): Articulation model with joints and bodies.
            q (ndarray): Generalized positions.

        Returns:
            list[Transform]: World transform per link.
    )pbdoc");

    m.def("inverse_dynamics", [](const Articulation& model,
                                 const VecXf& q, const VecXf& qd,
                                 const VecXf& qdd, const Vec3f& gravity) {
        return featherstone::inverse_dynamics(model, q, qd, qdd, gravity);
    }, py::arg("model"), py::arg("q"), py::arg("qd"),
       py::arg("qdd"), py::arg("gravity"),
    R"pbdoc(
        Runs Recursive Newton-Euler inverse dynamics.

        Args:
            model (Articulation): Articulation model.
            q (ndarray): Generalized positions.
            qd (ndarray): Generalized velocities.
            qdd (ndarray): Generalized accelerations.
            gravity (Vector3): Gravity vector in world coordinates (m/s^2).

        Returns:
            ndarray: Required joint efforts (torques/forces).
    )pbdoc");

    m.def("mass_matrix_crba", [](const Articulation& model, const VecXf& q) {
        return featherstone::mass_matrix(model, q);
    }, py::arg("model"), py::arg("q"),
    R"pbdoc(
        Computes the joint-space mass matrix using CRBA.

        Args:
            model (Articulation): Articulation model.
            q (ndarray): Generalized positions.

        Returns:
            ndarray: Symmetric positive-definite mass matrix H(q).
    )pbdoc");

    m.def("forward_dynamics", [](const Articulation& model,
                                 const VecXf& q, const VecXf& qd,
                                 const VecXf& tau, const Vec3f& gravity) {
        return featherstone::forward_dynamics(model, q, qd, tau, gravity);
    }, py::arg("model"), py::arg("q"), py::arg("qd"),
       py::arg("tau"), py::arg("gravity"),
    R"pbdoc(
        Computes generalized accelerations from applied joint efforts.

        Args:
            model (Articulation): Articulation model.
            q (ndarray): Generalized positions.
            qd (ndarray): Generalized velocities.
            tau (ndarray): Applied joint efforts.
            gravity (Vector3): Gravity vector in world coordinates (m/s^2).

        Returns:
            ndarray: Generalized accelerations.
    )pbdoc");

    m.def("forward_link_velocities", [](const Articulation& model,
                                        const VecXf& q, const VecXf& qd) {
        auto vs = featherstone::forward_link_velocities(model, q, qd);
        // Return as list of (angular, linear) tuples in link-local frame
        py::list result;
        for (const auto& v : vs) {
            result.append(py::make_tuple(spatial_angular(v), spatial_linear(v)));
        }
        return result;
    }, py::arg("model"), py::arg("q"), py::arg("qd"),
    R"pbdoc(
        Computes per-link spatial velocities from generalized state.

        Each element is a tuple (angular_velocity, linear_velocity) in
        the link-local frame, following the [angular; linear] convention.

        Args:
            model (Articulation): Articulation model.
            q (ndarray): Generalized positions.
            qd (ndarray): Generalized velocities.

        Returns:
            list[tuple[Vector3, Vector3]]: (omega, v_linear) per link in link-local frame.
    )pbdoc");

    // --- ArticulatedSolver ---
    py::class_<ArticulatedSolver>(m, "ArticulatedSolver", R"pbdoc(
        Stateful articulated-body integrator built on Featherstone dynamics.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Creates a solver with default integration settings.
        )pbdoc")
        .def("step", [](ArticulatedSolver& self, const Articulation& model,
                        VecXf q, VecXf qd, const VecXf& tau,
                        const Vec3f& gravity, float dt) {
            self.step(model, q, qd, tau, gravity, dt);
            return std::make_pair(q, qd);
        }, py::arg("model"), py::arg("q"), py::arg("qd"),
           py::arg("tau"), py::arg("gravity"), py::arg("dt"),
        R"pbdoc(
            Advances the articulated system by one time step.

            Args:
                model (Articulation): Articulation model.
                q (ndarray): Current generalized positions.
                qd (ndarray): Current generalized velocities.
                tau (ndarray): Applied joint efforts.
                gravity (Vector3): Gravity vector in world coordinates (m/s^2).
                dt (float): Integration time step in seconds.

            Returns:
                tuple[ndarray, ndarray]: Updated `(q, qd)` state.
        )pbdoc");

    py::class_<XPBDSolverSettings>(m, "XPBDSolverSettings", R"pbdoc(
        Runtime settings for the articulated XPBD solver scaffold.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("substeps", &XPBDSolverSettings::substeps)
        .def_readwrite("iterations", &XPBDSolverSettings::iterations)
        .def_readwrite("velocity_damping", &XPBDSolverSettings::velocity_damping)
        .def_readwrite("contact_relaxation", &XPBDSolverSettings::contact_relaxation)
        .def_readwrite("friction_damping", &XPBDSolverSettings::friction_damping);

    py::class_<XPBDStepStats>(m, "XPBDStepStats", R"pbdoc(
        Runtime statistics from the most recent XPBD step.
    )pbdoc")
        .def(py::init<>())
        .def_readonly("substeps", &XPBDStepStats::substeps)
        .def_readonly("iterations", &XPBDStepStats::iterations)
        .def_readonly("projected_constraints", &XPBDStepStats::projected_constraints)
        .def_readonly("contact_count", &XPBDStepStats::contact_count);

    py::class_<XPBDSolver>(m, "XPBDSolver", R"pbdoc(
        Reduced-coordinate XPBD solver scaffold for articulated systems.
    )pbdoc")
        .def(py::init<XPBDSolverSettings>(), py::arg("settings") = XPBDSolverSettings())
        .def_property("settings",
            [](XPBDSolver& self) -> XPBDSolverSettings& { return self.settings(); },
            [](XPBDSolver& self, const XPBDSolverSettings& settings) {
                self.settings() = settings;
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly("last_stats", &XPBDSolver::last_stats,
            py::return_value_policy::reference_internal)
        .def("step", [](XPBDSolver& self, const Articulation& model,
                         VecXf q, VecXf qd, const VecXf& tau,
                         const Vec3f& gravity, float dt,
                         const Control& control) {
            self.step(model, q, qd, tau, gravity, dt, control);
            return std::make_pair(q, qd);
        }, py::arg("model"), py::arg("q"), py::arg("qd"),
           py::arg("tau"), py::arg("gravity"), py::arg("dt"),
           py::arg("control") = Control())
        .def("step_with_contacts", [](XPBDSolver& self,
                                       const Articulation& model,
                                       const Model& collision_model,
                                       const std::vector<CollisionShape>& static_shapes,
                                       VecXf q,
                                       VecXf qd,
                                       const VecXf& tau,
                                       const Vec3f& gravity,
                                       float dt,
                                       const Control& control) {
            std::vector<ContactPoint> contacts;
            self.step_with_contacts(model, collision_model, static_shapes, q, qd, tau, gravity, dt, control, {}, &contacts);
            return py::make_tuple(q, qd, contacts);
        }, py::arg("model"), py::arg("collision_model"), py::arg("static_shapes"),
           py::arg("q"), py::arg("qd"), py::arg("tau"), py::arg("gravity"), py::arg("dt"),
           py::arg("control") = Control());
}
