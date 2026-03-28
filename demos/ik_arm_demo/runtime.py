"""Unified runtime for FR3 IK demo (with or without viewer).

Polyscope API differences across versions are handled via small ``_polyscope_try_*``
helpers instead of scattering version checks through the UI code.
"""

import importlib

import numpy as np
import novaphy
from novaphy.viz import make_sphere_mesh, quat_to_rotation_matrix

try:
    import polyscope as ps
    import polyscope.imgui as psim

    HAS_POLYSCOPE = True
except ImportError:
    ps = None
    psim = None
    HAS_POLYSCOPE = False

from .config import (
    ANIMATION_DT,
    DEFAULT_SMOOTH_ALPHA,
    DEFAULT_TARGET_POS,
    IK_HOLD_ERR_ENTER,
    IK_HOLD_ERR_EXIT,
    IK_HOLD_TARGET_DELTA,
    IK_MAX_ITER,
    IK_MIN_INTERVAL_S,
    IK_ARM_TARGET_SPHERE_RADIUS,
    R_IDENTITY,
)
from .ik_logic import (
    _as_vec3_f32,
    _as_vec3_f64,
    _solve_position_ik_goal,
    _tcp_target_world_from_gizmo,
    _vec3_changed,
    ee_world_xyz_m,
)
from .scene_setup import build_franka_ik_setup, build_franka_visual_parts


def _print_headless_ik_summary(converged, target_m_xyz, achieved_m_xyz, err_m):
    err_mm = 1000.0 * float(err_m)
    print(
        "IK headless: converged=%s\n"
        "  target TCP (world, m):  %10.5f %10.5f %10.5f\n"
        "  achieved EE (world, m): %10.5f %10.5f %10.5f\n"
        "  position error (mm):    %10.3f"
        % (
            converged,
            target_m_xyz[0],
            target_m_xyz[1],
            target_m_xyz[2],
            achieved_m_xyz[0],
            achieved_m_xyz[1],
            achieved_m_xyz[2],
            err_mm,
        )
    )


def _run_headless_ik_once(max_iter=None, target_pos=None, q_seed=None):
    setup = build_franka_ik_setup()
    art = setup["art"]
    ee_link_index = int(setup["ee_link_index"])
    nq = art.total_q()
    q_init = np.asarray(setup["q_init"], dtype=np.float32).ravel().copy()
    if q_seed is not None:
        qs = np.asarray(q_seed, dtype=np.float32).ravel()
        if qs.size == nq:
            q_init = qs.copy()

    joint_limits = setup["joint_limits"]
    p_tgt = np.asarray(DEFAULT_TARGET_POS if target_pos is None else target_pos, dtype=np.float64).ravel()
    iters = IK_MAX_ITER if max_iter is None else int(max_iter)
    q_sol, conv = _solve_position_ik_goal(
        art,
        q_init,
        p_tgt,
        joint_limits=joint_limits,
        ee_link_index=ee_link_index,
        max_iter=iters,
    )

    tgt_m = _as_vec3_f64(p_tgt, name="p_tgt")
    ee_m = ee_world_xyz_m(art, q_sol, ee_link_index=ee_link_index)
    err_pos = float(np.linalg.norm(ee_m - tgt_m))
    _print_headless_ik_summary(conv, tgt_m, ee_m, err_pos)
    return np.asarray(q_sol, dtype=np.float32), bool(conv), err_pos


def _slider_float3_to_xyz(slider_return, label="position"):
    if slider_return is None:
        return None
    if not isinstance(slider_return, (list, tuple)):
        return np.asarray(slider_return, dtype=np.float64).ravel()[:3]
    if len(slider_return) == 4:
        return np.array(
            [float(slider_return[1]), float(slider_return[2]), float(slider_return[3])],
            dtype=np.float64,
        )
    if len(slider_return) == 2:
        v = slider_return[1]
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size >= 3:
            return arr[:3].copy()
        raise ValueError("%s: expected 3 floats from SliderFloat3, got %r" % (label, slider_return))
    raise ValueError("%s: unexpected SliderFloat3 return %r" % (label, slider_return))


def _transform_4x4(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def _Rt_from_4x4(T):
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy().astype(np.float64)


def _polyscope_try_get_transform(structure):
    fn = getattr(structure, "get_transform", None)
    if callable(fn):
        try:
            M = np.asarray(fn(), dtype=np.float64)
            if M.shape == (4, 4):
                return _Rt_from_4x4(M)
        except (TypeError, ValueError, RuntimeError):
            pass
    return None, None


def _polyscope_try_set_transform(structure, R, t):
    T = _transform_4x4(R, t)
    fn = getattr(structure, "set_transform", None)
    if callable(fn):
        try:
            fn(T)
            return True
        except (TypeError, ValueError, RuntimeError):
            pass
    return False


def _polyscope_try_enable_transform_gizmo(structure, enabled=True):
    flag = bool(enabled)
    for name in (
        "set_transform_gizmo_enabled",
        "set_transform_enabled",
        "set_enable_transform",
        "set_transformable",
        "set_transform_handle_enabled",
    ):
        fn = getattr(structure, name, None)
        if callable(fn):
            try:
                fn(flag)
                return True
            except (TypeError, ValueError, RuntimeError):
                continue
    return False


def _update_target_gizmo_mesh(mesh_or_none, R_tgt, p_tgt, verts_local):
    R = np.asarray(R_tgt, dtype=np.float32)
    p = _as_vec3_f32(p_tgt, name="p_tgt")
    world_v = (verts_local @ R.T) + p.reshape(1, 3)
    if mesh_or_none is not None:
        mesh_or_none.update_vertex_positions(world_v)
    return world_v


def _sync_target_gizmo_world_position(gizmo_mesh, gizmo_verts, target_pos):
    """Apply Polyscope transform to the target gizmo, or rebuild vertex positions if unsupported."""
    if gizmo_mesh is None:
        return
    if _polyscope_try_set_transform(gizmo_mesh, R_IDENTITY, target_pos):
        return
    world_g = _update_target_gizmo_mesh(None, R_IDENTITY, target_pos, gizmo_verts)
    gizmo_mesh.update_vertex_positions(world_g)


def _setup_imgui_key_toggle():
    imgui_mod = None
    imgui_space_key = 32
    imgui_keypress_available = False
    try:
        _imgui = importlib.import_module("imgui")
        if hasattr(_imgui, "is_key_pressed"):
            imgui_mod = _imgui
            imgui_space_key = getattr(_imgui, "KEY_SPACE", 32)
            imgui_keypress_available = True
    except ImportError:
        pass
    return imgui_mod, imgui_space_key, imgui_keypress_available


def _should_toggle_play(imgui_mod, imgui_space_key: int, imgui_keypress_available: bool):
    if imgui_keypress_available and imgui_mod is not None:
        try:
            if imgui_mod.is_key_pressed(imgui_space_key):
                return True
        except (TypeError, RuntimeError, AttributeError):
            pass
    if hasattr(psim, "is_key_pressed"):
        try:
            k = getattr(psim, "KEY_SPACE", 32)
            return bool(psim.is_key_pressed(k))
        except (TypeError, RuntimeError, AttributeError):
            pass
    return False


def _ensure_target_gizmo(state, gizmo_verts, gizmo_faces):
    if state["gizmo_mesh"] is not None:
        return state["gizmo_mesh"]
    gizmo_mesh = ps.register_surface_mesh("ik_target", gizmo_verts.copy(), gizmo_faces)
    gizmo_mesh.set_color((1.0, 0.85, 0.1))
    _polyscope_try_enable_transform_gizmo(gizmo_mesh, True)
    _polyscope_try_set_transform(gizmo_mesh, R_IDENTITY, state["target_pos"])
    state["gizmo_mesh"] = gizmo_mesh
    return gizmo_mesh


def _handle_ui_input(state, gizmo_verts, gizmo_faces, imgui_mod, imgui_space_key, imgui_keypress_available):
    toggle_play = _should_toggle_play(imgui_mod, imgui_space_key, imgui_keypress_available)
    if psim.Button("Pause" if state["playing"] else "Play"):
        toggle_play = True
    if toggle_play:
        state["playing"] = not state["playing"]
    psim.SameLine()
    psim.Text("Space: toggle  |  %s" % ("PLAYING" if state["playing"] else "PAUSED"))

    psim.Separator()
    psim.TextUnformatted("Target: drag `ik_target` gizmo / slider — IK solves FR3 TCP position")
    target_before_ui = state["target_pos"].copy()
    out3 = psim.SliderFloat3(
        "position (m)",
        (
            float(target_before_ui[0]),
            float(target_before_ui[1]),
            float(target_before_ui[2]),
        ),
        -0.8,
        0.8,
    )
    target_from_slider = _slider_float3_to_xyz(out3, "position (m)")
    if target_from_slider is None:
        target_from_slider = target_before_ui.copy()
    _, state["smooth_alpha"] = psim.SliderFloat(
        "joint smooth alpha", float(state["smooth_alpha"]), 0.02, 1.0
    )

    gizmo_mesh = _ensure_target_gizmo(state, gizmo_verts, gizmo_faces)
    if _vec3_changed(target_from_slider, target_before_ui):
        state["target_pos"] = target_from_slider
        _sync_target_gizmo_world_position(gizmo_mesh, gizmo_verts, state["target_pos"])
        return

    _R_ps, t_ps = _polyscope_try_get_transform(gizmo_mesh)
    if t_ps is not None and _vec3_changed(t_ps, target_before_ui):
        state["target_pos"] = _as_vec3_f64(t_ps, name="gizmo_transform_translation")


def _update_ik_goal(state, art, ee_link_index, joint_limits):
    """Re-solve IK on a timer while playing, with hysteresis to avoid solution hopping.

    When the EE is already close to the IK goal and the user barely moves the target,
    ``ik_hold_active`` freezes re-solving until the error or target delta exceeds exit
    thresholds (see ``IK_HOLD_*`` in config).
    """
    if state["playing"]:
        state["sim_time"] += ANIMATION_DT
    if (not state["playing"]) or (state["sim_time"] - state["last_ik_time"]) < IK_MIN_INTERVAL_S:
        return

    state["last_ik_time"] = state["sim_time"]
    p_ik = _tcp_target_world_from_gizmo(
        art, state["q_display"], state["target_pos"], ee_link_index=ee_link_index
    )
    ee_m = ee_world_xyz_m(art, state["q_display"], ee_link_index=ee_link_index)
    ee_err = float(np.linalg.norm(ee_m - _as_vec3_f64(p_ik, name="p_ik")))
    target_delta = float(
        np.linalg.norm(_as_vec3_f64(state["target_pos"], name="target_pos") - state["last_ik_target_pos"])
    )

    if state["ik_hold_active"]:
        if ee_err > IK_HOLD_ERR_EXIT or target_delta > IK_HOLD_TARGET_DELTA:
            state["ik_hold_active"] = False
    elif ee_err < IK_HOLD_ERR_ENTER and target_delta < IK_HOLD_TARGET_DELTA:
        state["ik_hold_active"] = True
    if state["ik_hold_active"]:
        return

    state["q_goal"][:], state["ik_last_converged"] = _solve_position_ik_goal(
        art,
        state["q_display"],
        p_ik,
        joint_limits=joint_limits,
        ee_link_index=ee_link_index,
        max_iter=IK_MAX_ITER,
        prefer_q=state["q_display"],
    )
    state["last_ik_target_pos"] = _as_vec3_f64(state["target_pos"], name="target_pos").copy()


def _update_scene_meshes(art, q_display, visual_parts):
    def _apply_surface_render_style(mesh):
        if hasattr(mesh, "set_smooth_shade"):
            mesh.set_smooth_shade(True)
        # Some imported DAE meshes may contain mixed triangle winding.
        # If supported, shade back faces the same way to avoid tiny black patches.
        if hasattr(mesh, "set_back_face_policy"):
            for policy in ("identical", "Identical"):
                try:
                    mesh.set_back_face_policy(policy)
                    break
                except (TypeError, ValueError, RuntimeError, AttributeError):
                    continue

    transforms = novaphy.forward_kinematics(art, q_display)
    for spec in visual_parts:
        body_idx = int(spec["body_index"])
        if body_idx < 0 or body_idx >= len(transforms):
            continue
        pos = np.array(transforms[body_idx].position, dtype=np.float32)
        rot = quat_to_rotation_matrix(np.array(transforms[body_idx].rotation, dtype=np.float32))
        world_v = (spec["verts_local"] @ rot.T) + pos
        name = spec["name"]
        if ps.has_surface_mesh(name):
            ps.get_surface_mesh(name).update_vertex_positions(world_v)
        else:
            m = ps.register_surface_mesh(name, world_v, spec["faces"])
            m.set_color(spec["color"])
            _apply_surface_render_style(m)


def _update_runtime_markers(state, art, ee_link_index, gizmo_verts):
    ee_m = ee_world_xyz_m(art, state["q_display"], ee_link_index=ee_link_index)
    ee_pos = _as_vec3_f32(ee_m, name="ee_pos")
    p_ik_live = _tcp_target_world_from_gizmo(
        art, state["q_display"], state["target_pos"], ee_link_index=ee_link_index
    )
    state["last_pos_err"] = float(
        np.linalg.norm(ee_m - _as_vec3_f64(p_ik_live, name="p_ik_live"))
    )

    if ps.has_point_cloud("ee_tip"):
        ps.get_point_cloud("ee_tip").update_point_positions(ee_pos.reshape(1, 3))
    else:
        pc = ps.register_point_cloud("ee_tip", ee_pos.reshape(1, 3))
        pc.set_radius(0.00012, relative=False)
        pc.set_color((0.1, 0.9, 0.2))

    _sync_target_gizmo_world_position(state["gizmo_mesh"], gizmo_verts, state["target_pos"])


def _render_status_panel(state):
    psim.Separator()
    err_mm = 1000.0 * float(state["last_pos_err"])
    psim.Text("  TCP pos err (mm): %.3f" % err_mm)
    psim.Text(
        "IK: %s%s  |  smooth a=%.2f"
        % (
            "ok" if state["ik_last_converged"] else "running / local min",
            " | hold" if state["ik_hold_active"] else "",
            state["smooth_alpha"],
        )
    )


def _make_polyscope_frame_callback(
    state,
    art,
    ee_link_index,
    joint_limits,
    gizmo_verts,
    gizmo_faces,
    visual_parts,
    imgui_mod,
    imgui_space_key,
    imgui_keypress_available,
):
    """Per-frame UI, IK, smoothing, mesh FK, and overlays for the interactive demo."""

    def callback():
        _handle_ui_input(
            state,
            gizmo_verts,
            gizmo_faces,
            imgui_mod=imgui_mod,
            imgui_space_key=imgui_space_key,
            imgui_keypress_available=imgui_keypress_available,
        )
        _update_ik_goal(state, art, ee_link_index, joint_limits)
        if state["playing"]:
            a = float(np.clip(state["smooth_alpha"], 0.0, 1.0))
            state["q_display"] = (
                state["q_display"] + a * (state["q_goal"] - state["q_display"])
            ).astype(np.float32)
        _update_scene_meshes(art, state["q_display"], visual_parts)
        _update_runtime_markers(state, art, ee_link_index, gizmo_verts)
        _render_status_panel(state)

    return callback


def run_runtime(
    *,
    headless: bool = False,
    max_iter=None,
    target_pos=None,
    q_seed=None,
):
    """Run the IK runtime in one unified implementation."""
    if headless or not HAS_POLYSCOPE:
        return _run_headless_ik_once(max_iter=max_iter, target_pos=target_pos, q_seed=q_seed)

    ps.init()
    ps.set_program_name("NovaPhy - FR3 IK (position target)")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    setup = build_franka_ik_setup()
    art = setup["art"]
    ee_link_index = int(setup["ee_link_index"])
    joint_limits = setup["joint_limits"]
    q_display = np.asarray(setup["q_init"], dtype=np.float32).ravel().copy()
    target_pos = DEFAULT_TARGET_POS.astype(np.float64).copy()
    q_goal = q_display.copy()

    visual_parts = build_franka_visual_parts(
        art,
        ee_link_index,
        setup["scene"],
        scene_metadata=setup["scene"].metadata,
        urdf_model=setup["urdf_model"],
        urdf_path=setup["urdf_path"],
    )
    gizmo_verts, gizmo_faces = make_sphere_mesh(
        float(IK_ARM_TARGET_SPHERE_RADIUS), n_lat=12, n_lon=24
    )

    p_ik0 = _tcp_target_world_from_gizmo(
        art, q_display, target_pos, ee_link_index=ee_link_index
    )
    q_goal[:], ik_last_converged = _solve_position_ik_goal(
        art,
        q_display,
        p_ik0,
        joint_limits=joint_limits,
        ee_link_index=ee_link_index,
        max_iter=IK_MAX_ITER,
        prefer_q=q_display,
    )

    state = {
        "sim_time": 0.0,
        "playing": True,
        "q_display": q_display,
        "q_goal": q_goal,
        "target_pos": target_pos,
        "smooth_alpha": DEFAULT_SMOOTH_ALPHA,
        "last_ik_time": -1e9,
        "ik_last_converged": bool(ik_last_converged),
        "ik_hold_active": False,
        "last_ik_target_pos": target_pos.copy(),
        "last_pos_err": 0.0,
        "gizmo_mesh": None,
    }
    imgui_mod, imgui_space_key, imgui_keypress_available = _setup_imgui_key_toggle()
    callback = _make_polyscope_frame_callback(
        state,
        art,
        ee_link_index,
        joint_limits,
        gizmo_verts,
        gizmo_faces,
        visual_parts,
        imgui_mod,
        imgui_space_key,
        imgui_keypress_available,
    )

    ps.register_point_cloud("base_origin", np.zeros((1, 3), dtype=np.float32))
    ps.set_user_callback(callback)
    ps.show()
    return None


__all__ = ["run_runtime"]
