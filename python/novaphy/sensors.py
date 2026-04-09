"""Sensor framework for extracting observations from NovaPhy simulations.

Sensors follow an init-update-read pattern inspired by Newton Physics:
  1. Configure: sensor = SensorIMU(model, sites="imu_*")
  2. Update:    sensor.update(state, dt=dt)
  3. Read:      acc = sensor.accelerometer  # numpy (N,3)
"""

from __future__ import annotations
import fnmatch
from typing import Union

import numpy as np
import novaphy


def resolve_sites(
    model: novaphy.Model,
    sites: Union[str, list[str], list[int]],
) -> list[int]:
    """Resolve site specifier to a list of site indices.

    Accepts:
      - list of int indices: returned as-is
      - single str pattern: fnmatch against all site labels
      - list of str patterns: union of all matches
    """
    all_sites = model.sites
    if isinstance(sites, list) and len(sites) > 0 and isinstance(sites[0], int):
        return list(sites)
    if isinstance(sites, str):
        sites = [sites]
    indices = []
    for pattern in sites:
        for i, s in enumerate(all_sites):
            if fnmatch.fnmatch(s.label, pattern) and i not in indices:
                indices.append(i)
    return indices


def resolve_body_indices(
    model: novaphy.Model,
    bodies: Union[list[int], None],
) -> list[int]:
    """Resolve body specifier to a list of body indices."""
    if bodies is None:
        return list(range(model.num_bodies))
    if isinstance(bodies, list):
        return list(bodies)
    raise TypeError("body specifier must be list[int] or None (all bodies)")


def compute_site_world_transforms(
    model: novaphy.Model,
    state: novaphy.SimState,
) -> list[novaphy.Transform]:
    """Compute world-frame transforms for all sites in the model.

    For free-body sites: world_tf = body_transform * site.local_transform
    For articulation sites: world_tf = link_world_transform * site.local_transform
    """
    results = []
    art_fk_cache: dict[int, list] = {}
    for site in model.sites:
        if site.is_articulation_site():
            ai = site.articulation_index
            if ai not in art_fk_cache:
                art = model.articulations[ai]
                art_fk_cache[ai] = novaphy.forward_kinematics(art, state.q[ai])
            link_tf = art_fk_cache[ai][site.link_index]
            world_tf = link_tf * site.local_transform
        else:
            body_index = site.body_index
            if body_index < 0 or body_index >= model.num_bodies:
                raise ValueError(
                    f"Invalid non-articulation site attachment: body_index={body_index}, "
                    f"expected 0 <= body_index < {model.num_bodies}"
                )
            body_tf = state.transforms[body_index]
            world_tf = body_tf * site.local_transform
        results.append(world_tf)
    return results


class SensorIMU:
    """Inertial measurement unit sensor attached to sites.

    Measures linear acceleration (accelerometer) and angular velocity
    (gyroscope) in the sensor-local frame.

    Accelerometer reports specific force (proper acceleration), which
    equals coordinate acceleration minus gravity. At rest on a surface,
    reads +g upward. In free fall, reads zero.

    For free-body sites offset from the body origin, the site velocity
    includes the ``omega x r`` rotational contribution.

    For articulation sites, link spatial velocities are obtained via
    ``forward_link_velocities`` and then corrected for the site offset
    within the link frame.

    Args:
        model: Immutable simulation model with sites.
        sites: Site specifier (str pattern, list of patterns, or list of int indices).
    """

    def __init__(self, model: novaphy.Model, sites: Union[str, list[str], list[int]]):
        self._model = model
        self._site_indices = resolve_sites(model, sites)
        n = len(self._site_indices)
        self.accelerometer = np.zeros((n, 3), dtype=np.float32)
        self.gyroscope = np.zeros((n, 3), dtype=np.float32)
        self._prev_linear_vel: np.ndarray | None = None

    @property
    def num_sensors(self) -> int:
        return len(self._site_indices)

    def update(self, state: novaphy.SimState, dt: float,
               gravity: np.ndarray | None = None) -> None:
        """Compute IMU readings from current state.

        Args:
            state: Current simulation state after a step.
            dt: Time step used in the last simulation step (seconds).
            gravity: World gravity vector (3,). If None, uses model.gravity.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        if gravity is None:
            g = np.array(self._model.gravity, dtype=np.float32)
        else:
            g = np.asarray(gravity, dtype=np.float32)

        site_world_tfs = compute_site_world_transforms(self._model, state)

        cur_vel = np.zeros((self.num_sensors, 3), dtype=np.float32)

        # Cache link velocities and FK results per articulation to avoid redundant calls.
        art_link_vels: dict[int, list] = {}
        art_fk_cache: dict[int, list] = {}

        for i, si in enumerate(self._site_indices):
            site = self._model.sites[si]
            tf = site_world_tfs[si]
            R_inv = np.array(tf.rotation_matrix(), dtype=np.float32).T

            if site.is_articulation_site():
                ai = site.articulation_index
                if ai not in art_link_vels:
                    art = self._model.articulations[ai]
                    q = state.q[ai]
                    qd = state.qd[ai]
                    art_link_vels[ai] = novaphy.forward_link_velocities(art, q, qd)
                    art_fk_cache[ai] = novaphy.forward_kinematics(art, q)

                link_vel = art_link_vels[ai][site.link_index]
                omega_local = np.array(link_vel[0], dtype=np.float32)
                v_local = np.array(link_vel[1], dtype=np.float32)

                R_link = np.array(
                    art_fk_cache[ai][site.link_index].rotation_matrix(),
                    dtype=np.float32)

                omega_world = R_link @ omega_local

                r_site_local = np.array(site.local_transform.position, dtype=np.float32)
                v_site_local = v_local + np.cross(omega_local, r_site_local)
                vel_world = R_link @ v_site_local
            else:
                body_idx = site.body_index
                v_body = np.array(state.linear_velocities[body_idx],
                                  dtype=np.float32)
                omega_world = np.array(state.angular_velocities[body_idx],
                                       dtype=np.float32)

                body_pos = np.array(state.transforms[body_idx].position,
                                    dtype=np.float32)
                site_pos = np.array(tf.position, dtype=np.float32)
                r_offset = site_pos - body_pos
                vel_world = v_body + np.cross(omega_world, r_offset)

            self.gyroscope[i] = R_inv @ omega_world
            cur_vel[i] = vel_world

            if self._prev_linear_vel is not None:
                a_world = (vel_world - self._prev_linear_vel[i]) / dt
                self.accelerometer[i] = R_inv @ (a_world - g)
            else:
                self.accelerometer[i] = R_inv @ (-g)

        self._prev_linear_vel = cur_vel.copy()


class SensorContact:
    """Contact force sensor for monitoring forces on specified bodies.

    Aggregates solver-computed contact impulses into per-body force and
    torque readings in world frame.  Force = impulse / dt, torque is
    accumulated via r x F where r is the lever arm from body CoM to
    the contact point.

    This sensor currently assumes ``contacts`` come from the default
    free-body sequential-impulse pipeline, where each ``ContactPoint``
    carries a solver-populated ``contact_impulse`` for the current step.
    Other solver pipelines may expose contacts without this field being
    meaningfully populated yet.

    Args:
        model: Immutable simulation model.
        body_indices: List of body indices to monitor, or None for all bodies.
    """

    def __init__(self, model: novaphy.Model,
                 body_indices: Union[list[int], None] = None):
        self._model = model
        self._body_indices = resolve_body_indices(model, body_indices)
        n = len(self._body_indices)
        self.forces = np.zeros((n, 3), dtype=np.float32)
        self.torques = np.zeros((n, 3), dtype=np.float32)
        self.num_contacts = np.zeros(n, dtype=np.int32)

    @property
    def num_sensors(self) -> int:
        return len(self._body_indices)

    def update(self, state: novaphy.SimState,
               contacts: list, dt: float) -> None:
        """Compute per-body contact forces and torques from solver impulses.

        For each contact point, the solver provides ``contact_impulse``
        (world-frame total impulse vector).  Force = impulse / dt.
        Normal convention: impulse pushes body_b away from body_a, so
        body_a receives -F and body_b receives +F.

        Args:
            state: Current simulation state (for body transforms).
            contacts: Contact point list from world.contacts.
            dt: Time step used (seconds).
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.forces[:] = 0
        self.torques[:] = 0
        self.num_contacts[:] = 0

        body_to_sensor = {}
        for i, bi in enumerate(self._body_indices):
            body_to_sensor[bi] = i

        inv_dt = 1.0 / dt

        for cp in contacts:
            impulse = np.array(cp.contact_impulse, dtype=np.float32)
            force = impulse * inv_dt
            pos = np.array(cp.position, dtype=np.float32)

            if cp.body_a in body_to_sensor:
                si = body_to_sensor[cp.body_a]
                self.num_contacts[si] += 1
                f_a = -force
                self.forces[si] += f_a
                r_a = pos - np.array(state.transforms[cp.body_a].position,
                                     dtype=np.float32)
                self.torques[si] += np.cross(r_a, f_a)

            if cp.body_b in body_to_sensor:
                si = body_to_sensor[cp.body_b]
                self.num_contacts[si] += 1
                f_b = force
                self.forces[si] += f_b
                r_b = pos - np.array(state.transforms[cp.body_b].position,
                                     dtype=np.float32)
                self.torques[si] += np.cross(r_b, f_b)


class SensorFrameTransform:
    """Relative transform and velocity sensor between target and reference sites.

    Computes the transform of each target site expressed in the
    reference site's frame.  Optionally provides relative linear and
    angular velocities.

    If one reference site is given, it is broadcast to all targets.

    Args:
        model: Immutable simulation model.
        target_sites: Site specifier for target frames.
        reference_sites: Site specifier for reference frames.
    """

    def __init__(self, model: novaphy.Model,
                 target_sites: Union[str, list[str], list[int]],
                 reference_sites: Union[str, list[str], list[int]]):
        self._model = model
        self._target_indices = resolve_sites(model, target_sites)
        self._ref_indices = resolve_sites(model, reference_sites)
        n = len(self._target_indices)
        if len(self._ref_indices) == 1:
            self._ref_indices = self._ref_indices * n
        if len(self._ref_indices) != n:
            raise ValueError(
                f"reference_sites ({len(self._ref_indices)}) must be 1 or "
                f"match target_sites ({n})")
        self.positions = np.zeros((n, 3), dtype=np.float32)
        self.orientations = np.zeros((n, 4), dtype=np.float32)
        self.orientations[:, 3] = 1.0
        self.linear_velocities = np.zeros((n, 3), dtype=np.float32)
        self.angular_velocities = np.zeros((n, 3), dtype=np.float32)

    @property
    def num_sensors(self) -> int:
        return len(self._target_indices)

    def _site_world_velocity(self, state: novaphy.SimState,
                             site: novaphy.Site,
                             site_world_tf: novaphy.Transform,
                             art_link_vels: dict,
                             art_fk_cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear_vel, angular_vel) of a site in world frame."""
        if site.is_articulation_site():
            ai = site.articulation_index
            if ai not in art_link_vels:
                art = self._model.articulations[ai]
                q = state.q[ai]
                art_link_vels[ai] = novaphy.forward_link_velocities(
                    art, q, state.qd[ai])
                art_fk_cache[ai] = novaphy.forward_kinematics(art, q)
            link_vel = art_link_vels[ai][site.link_index]
            omega_local = np.array(link_vel[0], dtype=np.float32)
            v_local = np.array(link_vel[1], dtype=np.float32)

            R_link = np.array(
                art_fk_cache[ai][site.link_index].rotation_matrix(),
                dtype=np.float32)

            omega_world = R_link @ omega_local
            r_site_local = np.array(site.local_transform.position,
                                    dtype=np.float32)
            v_site_local = v_local + np.cross(omega_local, r_site_local)
            v_world = R_link @ v_site_local
            return v_world, omega_world
        else:
            bi = site.body_index
            v_body = np.array(state.linear_velocities[bi], dtype=np.float32)
            omega = np.array(state.angular_velocities[bi], dtype=np.float32)
            body_pos = np.array(state.transforms[bi].position,
                                dtype=np.float32)
            site_pos = np.array(site_world_tf.position, dtype=np.float32)
            r = site_pos - body_pos
            v_site = v_body + np.cross(omega, r)
            return v_site, omega

    def update(self, state: novaphy.SimState) -> None:
        """Compute relative transforms and velocities from current state."""
        world_tfs = compute_site_world_transforms(self._model, state)
        art_link_vels: dict[int, list] = {}
        art_fk_cache: dict[int, list] = {}

        for i, (ti, ri) in enumerate(
                zip(self._target_indices, self._ref_indices)):
            target_tf = world_tfs[ti]
            ref_tf = world_tfs[ri]

            ref_inv = ref_tf.inverse()
            rel = ref_inv * target_tf

            self.positions[i] = np.array(rel.position, dtype=np.float32)
            self.orientations[i] = np.array(rel.rotation, dtype=np.float32)

            # Relative velocity expressed in the reference frame.
            # Transport theorem: d/dt(R_ref^T (p_t-p_r)) = R_ref^T((v_t-v_r) - ω_r×r_rel)
            R_ref_inv = np.array(ref_tf.rotation_matrix(),
                                 dtype=np.float32).T

            target_site = self._model.sites[ti]
            ref_site = self._model.sites[ri]

            v_t, w_t = self._site_world_velocity(
                state, target_site, target_tf, art_link_vels, art_fk_cache)
            v_r, w_r = self._site_world_velocity(
                state, ref_site, ref_tf, art_link_vels, art_fk_cache)

            p_t = np.array(target_tf.position, dtype=np.float32)
            p_r = np.array(ref_tf.position, dtype=np.float32)
            r_rel_world = p_t - p_r

            self.linear_velocities[i] = R_ref_inv @ (
                (v_t - v_r) - np.cross(w_r, r_rel_world))
            self.angular_velocities[i] = R_ref_inv @ (w_t - w_r)
