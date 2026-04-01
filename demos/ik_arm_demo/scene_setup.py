"""Scene setup and visual mesh construction for the FR3 IK demo.

URDF import, Collada/STL loading, mesh cleanup, and Polyscope-ready part lists live
together here so file-level caches (``_DAE_*_CACHE``) stay coherent.
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import novaphy
from novaphy.viz import (
    make_box_mesh,
    make_cylinder_mesh,
    make_sphere_mesh,
    transform_vertices,
)

from .config import (
    ENV_IK_ARM_DAE_ALLOW_FULL_NODE_TRANSFORM,
    ENV_IK_ARM_DAE_APPLY_NODE_MATS,
    ENV_IK_ARM_DAE_AUTO_SCALE_ONLY,
    ENV_IK_ARM_DAE_FORCE_SCALE_ONLY,
    ENV_IK_ARM_VIS,
    FRANKA_ARM_JOINT_NAMES,
    FRANKA_ARM_Q_INIT,
    FRANKA_EE_LINK_NAME,
    FRANKA_URDF_PATH,
    JOINT_ANGLE_MAX,
    JOINT_ANGLE_MIN,
    parse_env_flag,
)


def _log_info(message: str):
    print(f"[demo_ik_arm] {message}")


def find_articulation_link_index_by_name(metadata, link_name: str) -> int:
    """Return link index for ``link_name`` in scene metadata, or ``-1`` if unknown."""
    names = list(getattr(metadata, "articulation_link_names", []))
    for i, n in enumerate(names):
        if n == link_name:
            return i
    return -1


def _build_joint_limits_from_metadata(art, metadata, q_init):
    nq = art.total_q()
    q_lo = np.full(nq, JOINT_ANGLE_MIN, dtype=np.float32)
    q_hi = np.full(nq, JOINT_ANGLE_MAX, dtype=np.float32)
    arm_set = set(FRANKA_ARM_JOINT_NAMES)
    for entry in getattr(metadata, "joints", []):
        if int(entry.q_start) < 0 or int(entry.num_q) <= 0:
            continue
        q_start = int(entry.q_start)
        num_q = int(entry.num_q)
        lo = float(entry.lower_limit)
        hi = float(entry.upper_limit)
        if not np.isfinite(lo):
            lo = JOINT_ANGLE_MIN
        if not np.isfinite(hi):
            hi = JOINT_ANGLE_MAX
        if entry.joint_name not in arm_set:
            # Keep non-arm DOFs fixed (e.g., finger sliders) so IK behavior remains arm-only.
            lo = hi = float(q_init[q_start])
        q_lo[q_start : q_start + num_q] = lo
        q_hi[q_start : q_start + num_q] = hi
    return q_lo, q_hi


def build_franka_ik_setup():
    """Build FR3 articulation from URDF: parse, import scene, seed q_init, joint limits.

    Arm joint angles are overwritten from ``FRANKA_ARM_Q_INIT`` where metadata allows;
    non-arm DOFs (e.g. gripper) keep limits collapsed to the initial pose so IK stays
    arm-only. EE link index prefers ``FRANKA_EE_LINK_NAME``, else last link.
    """
    urdf_path = os.path.abspath(FRANKA_URDF_PATH)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(
            "Franka URDF not found: %s. Expected resource folder under demos/data." % urdf_path
        )

    parser = novaphy.UrdfParser()
    urdf_model = parser.parse_file(urdf_path)
    options = novaphy.UrdfImportOptions()
    options.floating_base = False
    options.enable_self_collisions = False
    options.collapse_fixed_joints = False
    options.ignore_inertial_definitions = False
    scene = novaphy.SceneBuilderEngine().build_from_urdf(urdf_model, options)
    art = scene.articulation

    nq = art.total_q()
    q_init = np.asarray(scene.initial_q, dtype=np.float32).ravel().copy()
    if q_init.size != nq:
        q_init = np.zeros(nq, dtype=np.float32)

    # Inject known-good arm seed by joint name.
    for entry in scene.metadata.joints:
        if (
            entry.joint_name in FRANKA_ARM_Q_INIT
            and int(entry.q_start) >= 0
            and int(entry.num_q) >= 1
        ):
            qi = int(entry.q_start)
            qv = float(FRANKA_ARM_Q_INIT[entry.joint_name])
            lo = float(entry.lower_limit) if np.isfinite(float(entry.lower_limit)) else JOINT_ANGLE_MIN
            hi = float(entry.upper_limit) if np.isfinite(float(entry.upper_limit)) else JOINT_ANGLE_MAX
            q_init[qi] = np.float32(np.clip(qv, lo, hi))

    q_lo, q_hi = _build_joint_limits_from_metadata(art, scene.metadata, q_init)
    ee_link_index = find_articulation_link_index_by_name(scene.metadata, FRANKA_EE_LINK_NAME)
    if ee_link_index < 0:
        ee_link_index = art.num_links() - 1

    return {
        "scene": scene,
        "art": art,
        "urdf_model": urdf_model,
        "urdf_path": urdf_path,
        "q_init": q_init.astype(np.float32),
        "joint_limits": (q_lo.astype(np.float32), q_hi.astype(np.float32)),
        "ee_link_index": int(ee_link_index),
    }


def _transform_local_vertices(v, R=None, t=None):
    """Apply local rotation/translation to mesh vertices."""
    vv = np.asarray(v, dtype=np.float32).copy()
    if R is not None:
        RR = np.asarray(R, dtype=np.float32).reshape(3, 3)
        vv = vv @ RR.T
    if t is not None:
        tt = np.asarray(t, dtype=np.float32).ravel()[:3]
        vv += tt
    return vv


def _concat_mesh_parts(parts):
    """Merge many triangle meshes into one (verts, faces)."""
    if not parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    verts_all = []
    faces_all = []
    v_off = 0
    for v, f in parts:
        vv = np.asarray(v, dtype=np.float32)
        ff = np.asarray(f, dtype=np.int32)
        verts_all.append(vv)
        faces_all.append(ff + v_off)
        v_off += int(vv.shape[0])
    return np.vstack(verts_all), np.vstack(faces_all)


def _sanitize_rgb_color(
    color: tuple[float, float, float],
    *,
    fallback: tuple[float, float, float] = (0.82, 0.82, 0.84),
    min_luma: float = 0.10,
    min_channel: float = 0.06,
) -> tuple[float, float, float]:
    """Keep mesh colors away from unintended pure-black values."""
    c = np.asarray(color, dtype=np.float64).ravel()
    if c.size < 3 or (not np.isfinite(c[:3]).all()):
        return fallback
    rgb = np.clip(c[:3], 0.0, 1.0)
    luma = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    if luma < float(min_luma):
        rgb = rgb + (float(min_luma) - luma)
    rgb = np.maximum(rgb, float(min_channel))
    rgb = np.clip(rgb, 0.0, 1.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _sanitize_triangle_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    area_eps: float = 1e-12,
    rel_area_eps: float = 1e-8,
    dedup_quantize_eps: float = 5e-5,
    explode_faces: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop invalid/degenerate faces that can cause black flickering artifacts."""
    vv = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    ff = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if ff.size == 0 or vv.size == 0:
        return vv, ff

    def _deduplicate_faces_in_mesh(
        v_local: np.ndarray, f_local: np.ndarray, quantize_eps: float
    ) -> np.ndarray:
        if f_local.size == 0:
            return f_local
        kept = []
        seen_keys: set[
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
        ] = set()
        scale = float(max(quantize_eps, 1e-12))
        for tri_idx in f_local:
            tri = v_local[tri_idx]
            tri_q = np.rint(tri / scale).astype(np.int64)
            key = tuple(
                sorted(
                    (
                        (int(tri_q[0, 0]), int(tri_q[0, 1]), int(tri_q[0, 2])),
                        (int(tri_q[1, 0]), int(tri_q[1, 1]), int(tri_q[1, 2])),
                        (int(tri_q[2, 0]), int(tri_q[2, 1]), int(tri_q[2, 2])),
                    )
                )
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept.append(tri_idx)
        if not kept:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(kept, dtype=np.int32).reshape(-1, 3)

    def _orient_faces_consistently(v_local: np.ndarray, f_local: np.ndarray) -> np.ndarray:
        if f_local.shape[0] <= 1:
            return f_local

        edge_to_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for fi, (a, b, c) in enumerate(f_local):
            for u, v in ((a, b), (b, c), (c, a)):
                if u == v:
                    continue
                key = (int(min(u, v)), int(max(u, v)))
                sign = 1 if int(u) < int(v) else -1
                edge_to_faces.setdefault(key, []).append((fi, sign))

        flips = np.zeros(f_local.shape[0], dtype=bool)
        visited = np.zeros(f_local.shape[0], dtype=bool)
        face_neighbors: dict[int, list[tuple[int, bool]]] = {i: [] for i in range(f_local.shape[0])}
        for items in edge_to_faces.values():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                fi, si = items[i]
                for j in range(i + 1, len(items)):
                    fj, sj = items[j]
                    need_opposite_flip = si == sj
                    face_neighbors[fi].append((fj, need_opposite_flip))
                    face_neighbors[fj].append((fi, need_opposite_flip))

        for start in range(f_local.shape[0]):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            flips[start] = False
            qi = 0
            while qi < len(queue):
                cur = queue[qi]
                qi += 1
                for nei, opposite in face_neighbors.get(cur, []):
                    target_flip = (not flips[cur]) if opposite else flips[cur]
                    if not visited[nei]:
                        visited[nei] = True
                        flips[nei] = target_flip
                        queue.append(nei)

        out = f_local.copy()
        flip_ids = np.where(flips)[0]
        if flip_ids.size > 0:
            out[flip_ids] = out[flip_ids][:, [0, 2, 1]]

        if out.shape[0] > 0:
            tri = v_local[out]
            p0 = tri[:, 0].astype(np.float64, copy=False)
            p1 = tri[:, 1].astype(np.float64, copy=False)
            p2 = tri[:, 2].astype(np.float64, copy=False)
            n = np.cross(p1 - p0, p2 - p0)
            c = (p0 + p1 + p2) / 3.0
            center = np.mean(v_local.astype(np.float64, copy=False), axis=0)
            orientation_score = float(np.sum((c - center) * n))
            if orientation_score < 0.0:
                out = out[:, [0, 2, 1]]
        return out

    def _explode_faces(v_local: np.ndarray, f_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if f_local.size == 0:
            return v_local, f_local
        tri = v_local[f_local.reshape(-1)].reshape(-1, 3).astype(np.float32, copy=False)
        f_new = np.arange(tri.shape[0], dtype=np.int32).reshape(-1, 3)
        return tri, f_new

    n_vert = int(vv.shape[0])
    ok = np.ones(ff.shape[0], dtype=bool)
    ok &= (ff[:, 0] >= 0) & (ff[:, 1] >= 0) & (ff[:, 2] >= 0)
    ok &= (ff[:, 0] < n_vert) & (ff[:, 1] < n_vert) & (ff[:, 2] < n_vert)
    ok &= (ff[:, 0] != ff[:, 1]) & (ff[:, 0] != ff[:, 2]) & (ff[:, 1] != ff[:, 2])
    if np.any(ok):
        tri = ff[ok]
        p0 = vv[tri[:, 0]]
        p1 = vv[tri[:, 1]]
        p2 = vv[tri[:, 2]]
        finite = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
        idx_ok = np.where(ok)[0]
        ok[idx_ok[~finite]] = False
    if np.any(ok):
        tri = ff[ok]
        q0 = vv[tri[:, 0]].astype(np.float64, copy=False)
        q1 = vv[tri[:, 1]].astype(np.float64, copy=False)
        q2 = vv[tri[:, 2]].astype(np.float64, copy=False)
        area2 = np.linalg.norm(np.cross(q1 - q0, q2 - q0), axis=1)
        if np.isfinite(vv).all() and vv.shape[0] > 0:
            bb_min = np.min(vv.astype(np.float64, copy=False), axis=0)
            bb_max = np.max(vv.astype(np.float64, copy=False), axis=0)
            diag2 = float(np.linalg.norm(bb_max - bb_min) ** 2)
        else:
            diag2 = 0.0
        area_eps_eff = max(float(area_eps), float(rel_area_eps) * diag2)
        deg = area2 <= area_eps_eff
        idx_ok = np.where(ok)[0]
        ok[idx_ok[deg]] = False
    ff_clean = ff[ok].astype(np.int32, copy=False)
    ff_clean = _deduplicate_faces_in_mesh(vv, ff_clean, quantize_eps=float(dedup_quantize_eps))
    ff_clean = _orient_faces_consistently(vv, ff_clean)
    vv_out = vv
    ff_out = ff_clean.astype(np.int32, copy=False)
    if bool(explode_faces):
        vv_out, ff_out = _explode_faces(vv_out, ff_out)
    return vv_out.astype(np.float32, copy=False), ff_out.astype(np.int32, copy=False)


def _deduplicate_overlapping_faces_per_body(
    colored_parts: list[tuple[int, np.ndarray, np.ndarray, tuple[float, float, float]]],
    *,
    quantize_eps: float = 5e-5,
) -> list[tuple[int, np.ndarray, np.ndarray, tuple[float, float, float]]]:
    """
    Remove geometrically duplicated triangles on the same body.

    Overlapping coplanar faces (often from split material submeshes in DAE) can
    cause depth-fighting, which appears as black flickering stripes while moving.
    """
    if not colored_parts:
        return colored_parts

    parts_by_body: dict[int, list[int]] = {}
    for i, (body_idx, _v, _f, _c) in enumerate(colored_parts):
        parts_by_body.setdefault(int(body_idx), []).append(i)

    keep_masks = [
        np.zeros(np.asarray(f, dtype=np.int32).shape[0], dtype=bool)
        for (_b, _v, f, _c) in colored_parts
    ]

    for _body_idx, part_indices in parts_by_body.items():
        best_by_key: dict[
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
            tuple[int, int, float],
        ] = {}
        for part_i in part_indices:
            _b, verts, faces, color = colored_parts[part_i]
            vv = np.asarray(verts, dtype=np.float32)
            ff = np.asarray(faces, dtype=np.int32)
            if ff.size == 0:
                continue
            luma = float(0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])
            for face_i in range(ff.shape[0]):
                tri_idx = ff[face_i]
                tri = vv[tri_idx]
                tri_q = np.rint(tri / float(quantize_eps)).astype(np.int64)
                key = tuple(
                    sorted(
                        (
                            (int(tri_q[0, 0]), int(tri_q[0, 1]), int(tri_q[0, 2])),
                            (int(tri_q[1, 0]), int(tri_q[1, 1]), int(tri_q[1, 2])),
                            (int(tri_q[2, 0]), int(tri_q[2, 1]), int(tri_q[2, 2])),
                        )
                    )
                )
                prev = best_by_key.get(key)
                if prev is None or luma > (prev[2] + 1e-9):
                    best_by_key[key] = (part_i, face_i, luma)

        for part_i, face_i, _luma in best_by_key.values():
            keep_masks[part_i][face_i] = True

    deduped: list[tuple[int, np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    for i, (body_idx, verts, faces, color) in enumerate(colored_parts):
        mask = keep_masks[i]
        ff = np.asarray(faces, dtype=np.int32)
        if ff.size == 0 or not np.any(mask):
            continue
        deduped.append((body_idx, np.asarray(verts, dtype=np.float32), ff[mask], color))
    return deduped


_DAE_TRIMESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DAE_SUBMESH_CACHE: dict[str, list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]] = {}
_COLLADA_PARSE_ORDER_CACHE: dict[str, str] = {}


def _homogeneous_affine_error(M: np.ndarray) -> float:
    Mm = np.asarray(M, dtype=np.float64)
    if Mm.shape != (4, 4):
        return float("inf")
    return float(np.linalg.norm(Mm[3, :] - np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))


def _parse_collada_matrix4x4(matrix_text: str, source_key: str | None = None) -> np.ndarray | None:
    vals = [v for v in matrix_text.replace("\n", " ").split(" ") if v.strip()]
    if len(vals) != 16:
        return None
    arr = np.array([float(v) for v in vals], dtype=np.float32)
    M_row_major = arr.reshape((4, 4), order="C")
    M_col_major = arr.reshape((4, 4), order="F")
    if source_key:
        cached = _COLLADA_PARSE_ORDER_CACHE.get(source_key, None)
        if cached == "C":
            return M_row_major
        if cached == "F":
            return M_col_major
    row_err = _homogeneous_affine_error(M_row_major)
    col_err = _homogeneous_affine_error(M_col_major)
    if row_err <= col_err:
        if source_key:
            _COLLADA_PARSE_ORDER_CACHE[source_key] = "C"
        return M_row_major
    if source_key:
        _COLLADA_PARSE_ORDER_CACHE[source_key] = "F"
    return M_col_major


def _matrix_uniform_scale(M: np.ndarray) -> float | None:
    A = np.asarray(M[:3, :3], dtype=np.float64)
    cn = np.array([np.linalg.norm(A[:, i]) for i in range(3)], dtype=np.float64)
    s = float(np.mean(cn))
    if s <= 0.0:
        return None
    if np.max(np.abs(cn - s)) > max(1e-8, 2e-2 * s):
        return None
    return s


def _matrix_has_nontrivial_translation(M: np.ndarray, tol: float = 1e-7) -> bool:
    t = np.asarray(M[:3, 3], dtype=np.float64)
    return float(np.linalg.norm(t)) > float(tol)


def _matrix_is_affine_homogeneous(M: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(
        np.max(
            np.abs(
                np.asarray(M[3, :], dtype=np.float64)
                - np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            )
        )
        <= float(tol)
    )


def _matrix_is_uniform_scaled_rotation(M: np.ndarray, tol: float = 2e-2) -> bool:
    s = _matrix_uniform_scale(M)
    if s is None or s <= 0.0:
        return False
    A = np.asarray(M[:3, :3], dtype=np.float64)
    R = A / s
    RtR = R.T @ R
    return bool(np.max(np.abs(RtR - np.eye(3, dtype=np.float64))) <= float(tol))


def _select_collada_node_transform(
    M: np.ndarray,
    *,
    force_scale_only: bool,
    auto_scale_only: bool,
    allow_full_node_transform: bool,
) -> np.ndarray:
    M_use = np.asarray(M, dtype=np.float32)
    s = _matrix_uniform_scale(M_use)
    if force_scale_only:
        if s is None:
            return M_use
        Ms = np.eye(4, dtype=np.float32)
        Ms[0, 0] = float(s)
        Ms[1, 1] = float(s)
        Ms[2, 2] = float(s)
        return Ms
    if allow_full_node_transform or not auto_scale_only:
        return M_use
    if s is None:
        return M_use
    if _matrix_is_affine_homogeneous(M_use):
        return M_use
    if _matrix_is_uniform_scaled_rotation(M_use) and not _matrix_has_nontrivial_translation(M_use):
        return M_use
    Ms = np.eye(4, dtype=np.float32)
    Ms[0, 0] = float(s)
    Ms[1, 1] = float(s)
    Ms[2, 2] = float(s)
    return Ms


def _load_collada_dae_as_trimesh(dae_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Collada (.dae) triangle mesh as (verts, faces)."""
    dae_path = os.path.abspath(dae_path)
    if dae_path in _DAE_TRIMESH_CACHE:
        return _DAE_TRIMESH_CACHE[dae_path]

    tree = ET.parse(dae_path)
    root = tree.getroot()
    ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"c": ns_uri} if ns_uri else {}

    def _findall(path: str, elem):
        return elem.findall(path, ns)

    def _parse_collada_matrix4x4_local(matrix_text: str) -> np.ndarray | None:
        return _parse_collada_matrix4x4(matrix_text, source_key=dae_path)

    def _transform_positions_by_4x4(positions: np.ndarray, M_col_major: np.ndarray) -> np.ndarray:
        ones = np.ones((positions.shape[0], 1), dtype=np.float32)
        p_h = np.concatenate([positions.astype(np.float32, copy=False), ones], axis=1)
        out = p_h @ M_col_major.T
        return out[:, :3]

    apply_node_mats = parse_env_flag(ENV_IK_ARM_DAE_APPLY_NODE_MATS, default=True)
    force_scale_only = parse_env_flag(ENV_IK_ARM_DAE_FORCE_SCALE_ONLY, default=False)
    auto_scale_only = parse_env_flag(ENV_IK_ARM_DAE_AUTO_SCALE_ONLY, default=True)
    allow_full_node_transform = parse_env_flag(
        ENV_IK_ARM_DAE_ALLOW_FULL_NODE_TRANSFORM, default=False
    )

    geometry_id_to_node_mats: dict[str, list[np.ndarray]] = {}
    if apply_node_mats:
        visual_scenes = _findall(".//c:visual_scene", root) if ns else root.findall(".//visual_scene")

        def _walk_node(node, parent_M: np.ndarray):
            mat_el = (
                node.find('c:matrix[@sid="transform"]', ns)
                if ns
                else node.find('matrix[@sid="transform"]')
            )
            if mat_el is None:
                mat_el = node.find("c:matrix", ns) if ns else node.find("matrix")
            M_local = np.eye(4, dtype=np.float32)
            if mat_el is not None and (mat_el.text or "").strip():
                M_parsed = _parse_collada_matrix4x4_local(mat_el.text or "")
                if M_parsed is not None:
                    M_local = M_parsed
            M_world = np.asarray(parent_M, dtype=np.float32) @ np.asarray(M_local, dtype=np.float32)
            insts = node.findall("c:instance_geometry", ns) if ns else node.findall("instance_geometry")
            for inst in insts:
                url = inst.attrib.get("url", "")
                if url.startswith("#"):
                    geometry_id = url[1:]
                    geometry_id_to_node_mats.setdefault(geometry_id, []).append(M_world)
            child_nodes = node.findall("c:node", ns) if ns else node.findall("node")
            for child in child_nodes:
                _walk_node(child, M_world)

        if visual_scenes:
            for vs in visual_scenes:
                top_nodes = vs.findall("c:node", ns) if ns else vs.findall("node")
                for top in top_nodes:
                    _walk_node(top, np.eye(4, dtype=np.float32))
        else:
            top_nodes = root.findall("c:node", ns) if ns else root.findall("node")
            node_elems = top_nodes if top_nodes else (_findall(".//c:node", root) if ns else root.findall(".//node"))
            for node in node_elems:
                _walk_node(node, np.eye(4, dtype=np.float32))

    source_floats_by_id: dict[str, np.ndarray] = {}
    source_elems = _findall(".//c:source", root) if ns else root.findall(".//source")
    for src in source_elems:
        fa = src.find("c:float_array", ns) if ns else src.find("float_array")
        if fa is None:
            continue
        src_id = src.attrib.get("id")
        float_id = fa.attrib.get("id")
        fa_text = (fa.text or "").strip()
        if (not src_id and not float_id) or not fa_text:
            continue
        arr = np.fromstring(fa_text, sep=" ", dtype=np.float32)
        if src_id:
            source_floats_by_id[src_id] = arr
        if float_id and float_id not in source_floats_by_id:
            source_floats_by_id[float_id] = arr

    verts_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    v_off = 0

    geometry_elems = _findall(".//c:geometry", root) if ns else root.findall(".//geometry")
    for geometry in geometry_elems:
        geometry_id = geometry.attrib.get("id", "")
        mesh = geometry.find("c:mesh", ns) if ns else geometry.find("mesh")
        if mesh is None:
            continue
        mats = (
            geometry_id_to_node_mats.get(geometry_id, [np.eye(4, dtype=np.float32)])
            if apply_node_mats
            else [np.eye(4, dtype=np.float32)]
        )
        vertices_pos_source: dict[str, str] = {}
        vert_elems = _findall(".//c:vertices", mesh) if ns else mesh.findall(".//vertices")
        for vert in vert_elems:
            vert_id = vert.attrib.get("id")
            if not vert_id:
                continue
            pos_inp = (
                vert.find('c:input[@semantic="POSITION"]', ns)
                if ns
                else vert.find('input[@semantic="POSITION"]')
            )
            if pos_inp is None:
                continue
            src_ref = pos_inp.attrib.get("source", "")
            if src_ref.startswith("#"):
                src_ref = src_ref[1:]
            if src_ref:
                vertices_pos_source[vert_id] = src_ref

        tri_elems = _findall(".//c:triangles", mesh) if ns else mesh.findall(".//triangles")
        for tri in tri_elems:
            tri_count = int(tri.attrib.get("count", "0"))
            if tri_count <= 0:
                continue
            inputs = _findall("c:input", tri) if ns else tri.findall("input")
            if not inputs:
                continue
            vertex_input = None
            for inp in inputs:
                if inp.attrib.get("semantic") == "VERTEX":
                    vertex_input = inp
                    break
            if vertex_input is None:
                continue
            verts_source_ref = vertex_input.attrib.get("source", "")
            if verts_source_ref.startswith("#"):
                verts_source_ref = verts_source_ref[1:]
            pos_source_id = vertices_pos_source.get(verts_source_ref)
            if not pos_source_id:
                continue
            pos_flat = source_floats_by_id.get(pos_source_id)
            if pos_flat is None or pos_flat.size < 3:
                continue
            positions = pos_flat.reshape(-1, 3)

            offsets_with_inps = [(int(inp.attrib.get("offset", "0")), inp) for inp in inputs]
            offsets_with_inps.sort(key=lambda x: x[0])
            n_inputs = len(offsets_with_inps) if offsets_with_inps else 1
            vertex_offset = int(vertex_input.attrib.get("offset", "0"))
            vertex_col = next(
                (i for i, (off, _inp) in enumerate(offsets_with_inps) if off == vertex_offset),
                0,
            )
            p_el = tri.find("c:p", ns) if ns else tri.find("p")
            if p_el is None:
                continue
            p_text = (p_el.text or "").strip()
            if not p_text:
                continue
            idx = np.fromstring(p_text, sep=" ", dtype=np.int64)
            if idx.size == tri_count * 3 * n_inputs:
                pass
            elif (tri_count * 3) > 0 and (idx.size % (tri_count * 3) == 0):
                n_inputs = int(idx.size // (tri_count * 3))
                vertex_col = int(vertex_col) if int(vertex_col) < n_inputs else 0
            idx = idx.reshape(tri_count * 3, n_inputs)
            vert_indices = idx[:, vertex_col].astype(np.int32)
            faces = vert_indices.reshape(tri_count, 3)

            for M in mats:
                M_use = _select_collada_node_transform(
                    M,
                    force_scale_only=force_scale_only,
                    auto_scale_only=auto_scale_only,
                    allow_full_node_transform=allow_full_node_transform,
                )
                positions_t = _transform_positions_by_4x4(positions, M_use)
                verts_all.append(positions_t)
                faces_all.append(faces + v_off)
                v_off += positions_t.shape[0]

    if not verts_all or not faces_all:
        raise ValueError(f"Failed to load any triangles from DAE: {dae_path}")
    verts = np.vstack(verts_all).astype(np.float32)
    faces = np.vstack(faces_all).astype(np.int32)
    _DAE_TRIMESH_CACHE[dae_path] = (verts, faces)
    return verts, faces


def _load_collada_dae_submeshes(
    dae_path: str,
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]:
    """Load Collada(.dae) as per-material colored submeshes."""
    dae_path = os.path.abspath(dae_path)
    if dae_path in _DAE_SUBMESH_CACHE:
        return _DAE_SUBMESH_CACHE[dae_path]

    tree = ET.parse(dae_path)
    root = tree.getroot()
    ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"c": ns_uri} if ns_uri else {}

    def _fa(path, elem):
        return elem.findall(path, ns) if ns else elem.findall(path.replace("c:", ""))

    def _f1(path, elem):
        return elem.find(path, ns) if ns else elem.find(path.replace("c:", ""))

    def _parse_mat4(text: str) -> np.ndarray | None:
        return _parse_collada_matrix4x4(text, source_key=dae_path)

    def _xform_by_mat4(positions, M):
        ones = np.ones((positions.shape[0], 1), dtype=np.float32)
        p_h = np.concatenate([positions.astype(np.float32, copy=False), ones], axis=1)
        return (p_h @ M.T)[:, :3]

    # 1) effect_id -> diffuse color
    effect_colors: dict[str, tuple[float, float, float]] = {}
    for eff in _fa(".//c:effect", root):
        eff_id = eff.attrib.get("id", "")
        if not eff_id:
            continue
        for tname in ("phong", "lambert", "blinn"):
            tag = f"c:{tname}" if ns else tname
            techs = _fa(f".//{tag}", eff)
            if not techs:
                continue
            dc = _f1("c:diffuse/c:color" if ns else "diffuse/color", techs[0])
            if dc is not None and (dc.text or "").strip():
                vals = dc.text.strip().split()
                if len(vals) >= 3:
                    effect_colors[eff_id] = (float(vals[0]), float(vals[1]), float(vals[2]))
            break

    # 2) material_id -> color
    material_colors: dict[str, tuple[float, float, float]] = {}
    for mat in _fa(".//c:material", root):
        mat_id = mat.attrib.get("id", "")
        ie = _f1("c:instance_effect" if ns else "instance_effect", mat)
        if ie is None:
            continue
        url = ie.attrib.get("url", "").lstrip("#")
        if url in effect_colors:
            material_colors[mat_id] = effect_colors[url]

    # 3) node matrices & symbol bindings
    geometry_bindings: dict[str, dict[str, str]] = {}
    geometry_node_mats: dict[str, list[np.ndarray]] = {}
    apply_node_mats = parse_env_flag(ENV_IK_ARM_DAE_APPLY_NODE_MATS, default=True)
    force_scale_only = parse_env_flag(ENV_IK_ARM_DAE_FORCE_SCALE_ONLY, default=False)
    auto_scale_only = parse_env_flag(ENV_IK_ARM_DAE_AUTO_SCALE_ONLY, default=True)
    allow_full = parse_env_flag(ENV_IK_ARM_DAE_ALLOW_FULL_NODE_TRANSFORM, default=False)
    if apply_node_mats:
        visual_scenes = _fa(".//c:visual_scene", root)

        def _walk_node(node, parent_M: np.ndarray):
            mel = (
                node.find('c:matrix[@sid="transform"]', ns)
                if ns
                else node.find('matrix[@sid="transform"]')
            )
            if mel is None:
                mel = node.find("c:matrix", ns) if ns else node.find("matrix")
            M_local = np.eye(4, dtype=np.float32)
            if mel is not None and (mel.text or "").strip():
                Mp = _parse_mat4(mel.text or "")
                if Mp is not None:
                    M_local = Mp
            M_world = np.asarray(parent_M, dtype=np.float32) @ np.asarray(M_local, dtype=np.float32)
            insts = node.findall("c:instance_geometry", ns) if ns else node.findall("instance_geometry")
            for inst in insts:
                gid = inst.attrib.get("url", "").lstrip("#")
                if gid:
                    geometry_node_mats.setdefault(gid, []).append(M_world)
                for bm in (
                    inst.findall(".//c:instance_material", ns)
                    if ns
                    else inst.findall(".//instance_material")
                ):
                    sym = bm.attrib.get("symbol", "")
                    tgt = bm.attrib.get("target", "").lstrip("#")
                    if gid and sym and tgt:
                        geometry_bindings.setdefault(gid, {})[sym] = tgt
            child_nodes = node.findall("c:node", ns) if ns else node.findall("node")
            for child in child_nodes:
                _walk_node(child, M_world)

        if visual_scenes:
            for vs in visual_scenes:
                top_nodes = vs.findall("c:node", ns) if ns else vs.findall("node")
                for top in top_nodes:
                    _walk_node(top, np.eye(4, dtype=np.float32))
        else:
            top_nodes = root.findall("c:node", ns) if ns else root.findall("node")
            node_elems = top_nodes if top_nodes else _fa(".//c:node", root)
            for node in node_elems:
                _walk_node(node, np.eye(4, dtype=np.float32))

    # 4) parse source arrays
    source_floats: dict[str, np.ndarray] = {}
    for src in _fa(".//c:source", root):
        fa = _f1("c:float_array" if ns else "float_array", src)
        if fa is None:
            continue
        sid = src.attrib.get("id")
        fid = fa.attrib.get("id")
        txt = (fa.text or "").strip()
        if (not sid and not fid) or not txt:
            continue
        arr = np.fromstring(txt, sep=" ", dtype=np.float32)
        if sid:
            source_floats[sid] = arr
        if fid and fid not in source_floats:
            source_floats[fid] = arr

    name_rgb_re = re.compile(r"color_(\d+)_(\d+)_(\d+)")

    def _color_for(symbol: str, geom_id: str) -> tuple[float, float, float]:
        for name in (symbol, geometry_bindings.get(geom_id, {}).get(symbol, "")):
            m = name_rgb_re.search(name)
            if m:
                return _sanitize_rgb_color(
                    (
                        int(m.group(1)) / 255.0,
                        int(m.group(2)) / 255.0,
                        int(m.group(3)) / 255.0,
                    )
                )
        bound = geometry_bindings.get(geom_id, {}).get(symbol, symbol)
        if bound in material_colors:
            return _sanitize_rgb_color(material_colors[bound])
        if symbol in material_colors:
            return _sanitize_rgb_color(material_colors[symbol])
        return _sanitize_rgb_color((0.8, 0.8, 0.8))

    submeshes: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    for geometry in _fa(".//c:geometry", root):
        gid = geometry.attrib.get("id", "")
        mesh = _f1("c:mesh" if ns else "mesh", geometry)
        if mesh is None:
            continue
        mats = (
            geometry_node_mats.get(gid, [np.eye(4, dtype=np.float32)])
            if apply_node_mats
            else [np.eye(4, dtype=np.float32)]
        )
        vertices_pos_source: dict[str, str] = {}
        for vert in (_fa(".//c:vertices", mesh) if ns else mesh.findall(".//vertices")):
            vid = vert.attrib.get("id")
            if not vid:
                continue
            pin = (
                vert.find('c:input[@semantic="POSITION"]', ns)
                if ns
                else vert.find('input[@semantic="POSITION"]')
            )
            if pin is None:
                continue
            src = pin.attrib.get("source", "").lstrip("#")
            if src:
                vertices_pos_source[vid] = src

        tri_elems = _fa(".//c:triangles", mesh) if ns else mesh.findall(".//triangles")
        for tri in tri_elems:
            tri_count = int(tri.attrib.get("count", "0"))
            if tri_count <= 0:
                continue
            mat_sym = tri.attrib.get("material", "")
            color = _color_for(mat_sym, gid)
            inputs = _fa("c:input", tri) if ns else tri.findall("input")
            if not inputs:
                continue
            vertex_input = None
            for inp in inputs:
                if inp.attrib.get("semantic") == "VERTEX":
                    vertex_input = inp
                    break
            if vertex_input is None:
                continue
            verts_source_ref = vertex_input.attrib.get("source", "").lstrip("#")
            pos_source_id = vertices_pos_source.get(verts_source_ref)
            if not pos_source_id:
                continue
            pos_flat = source_floats.get(pos_source_id)
            if pos_flat is None or pos_flat.size < 3:
                continue
            positions = pos_flat.reshape(-1, 3)

            offsets_with_inps = [(int(inp.attrib.get("offset", "0")), inp) for inp in inputs]
            offsets_with_inps.sort(key=lambda x: x[0])
            n_inputs = len(offsets_with_inps) if offsets_with_inps else 1
            vertex_offset = int(vertex_input.attrib.get("offset", "0"))
            vertex_col = next(
                (i for i, (off, _inp) in enumerate(offsets_with_inps) if off == vertex_offset),
                0,
            )
            p_el = tri.find("c:p", ns) if ns else tri.find("p")
            if p_el is None or not (p_el.text or "").strip():
                continue
            idx = np.fromstring((p_el.text or "").strip(), sep=" ", dtype=np.int64)
            if idx.size == tri_count * 3 * n_inputs:
                pass
            elif (tri_count * 3) > 0 and (idx.size % (tri_count * 3) == 0):
                n_inputs = int(idx.size // (tri_count * 3))
                vertex_col = int(vertex_col) if int(vertex_col) < n_inputs else 0
            idx = idx.reshape(tri_count * 3, n_inputs)
            faces = idx[:, vertex_col].astype(np.int32).reshape(tri_count, 3)

            for M in mats:
                M_use = _select_collada_node_transform(
                    M,
                    force_scale_only=force_scale_only,
                    auto_scale_only=auto_scale_only,
                    allow_full_node_transform=allow_full,
                )
                pos_t = _xform_by_mat4(positions, M_use)
                v_clean, f_clean = _sanitize_triangle_mesh(pos_t, faces)
                if int(f_clean.shape[0]) <= 0:
                    continue
                submeshes.append(
                    (
                        v_clean.astype(np.float32, copy=False),
                        f_clean.astype(np.int32, copy=False),
                        _sanitize_rgb_color(color),
                    )
                )

    if not submeshes:
        raise ValueError(f"No triangles in DAE: {dae_path}")

    _DAE_SUBMESH_CACHE[dae_path] = submeshes
    return submeshes


def _cylinder_along_neg_z_from_origin(length, radius, n_segments=24):
    """Cylinder in link frame with axis along -Z, z range [-length, 0]."""
    half_len = 0.5 * float(length)
    v, f = make_cylinder_mesh(float(radius), half_len, n_segments=n_segments)
    v = np.asarray(v, dtype=np.float32)
    v[:, 2] -= half_len
    return v, f


def _rotation_from_z_axis(direction):
    """Rotation matrix that maps +Z onto `direction`."""
    d = np.asarray(direction, dtype=np.float64).ravel()[:3]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.eye(3, dtype=np.float32)
    d = d / n
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float32)
    if c < -1.0 + 1e-9:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    v = np.cross(z, d)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return np.asarray(R, dtype=np.float32)


def _cylinder_between_local_points(p0, p1, radius, n_segments=20):
    p0 = np.asarray(p0, dtype=np.float32).ravel()[:3]
    p1 = np.asarray(p1, dtype=np.float32).ravel()[:3]
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        v, f = make_sphere_mesh(float(radius), n_lat=8, n_lon=16)
        v = np.asarray(v, dtype=np.float32) + p0.reshape(1, 3)
        return v, f
    v, f = make_cylinder_mesh(float(radius), 0.5 * L, n_segments=n_segments)
    R = _rotation_from_z_axis(d)
    c = 0.5 * (p0 + p1)
    v = _transform_local_vertices(v, R=R, t=c)
    return v, f


def _ik_arm_visual_mode() -> str:
    """Return ``mesh`` (default), ``dae``, or ``collision`` from env."""
    v = os.environ.get(ENV_IK_ARM_VIS, "mesh").strip().lower()
    if v in ("mesh", "urdf_mesh", "visual", ""):
        return "mesh"
    if v in ("dae", "urdf_visual"):
        return "dae"
    if v in ("collision", "shapes", "primitive", "primitives"):
        return "collision"
    if v in ("1", "true", "yes"):
        return "mesh"
    _log_info(f"unknown {ENV_IK_ARM_VIS}={v!r}; using mesh")
    return "mesh"


def _fr3_component_color(body_idx: int, link_name: str | None = None) -> tuple[float, float, float]:
    by_name = {
        "base": (0.90, 0.96, 0.96),
        "fr3_link0": (1.00, 0.42, 0.77),
        "fr3_link1": (0.86, 0.29, 0.29),
        "fr3_link2": (0.94, 0.57, 0.19),
        "fr3_link3": (0.94, 0.82, 0.20),
        "fr3_link4": (0.30, 0.70, 0.34),
        "fr3_link5": (0.22, 0.66, 0.80),
        "fr3_link6": (0.25, 0.44, 0.86),
        "fr3_link7": (0.55, 0.42, 0.86),
        "fr3_link8": (0.72, 0.42, 0.74),
        "fr3_hand": (0.4, 0.8, 1.0),
        "fr3_leftfinger": (0.85, 0.90, 1.00),
        "fr3_rightfinger": (0.85, 0.90, 1.00),
        "fr3_hand_tcp": (0.89, 0.26, 0.20),
    }
    if link_name is not None and link_name in by_name:
        return by_name[link_name]
    fallback = [
        (0.86, 0.29, 0.29),
        (0.94, 0.57, 0.19),
        (0.94, 0.82, 0.20),
        (0.30, 0.70, 0.34),
        (0.22, 0.66, 0.80),
        (0.25, 0.44, 0.86),
        (0.55, 0.42, 0.86),
    ]
    return fallback[int(body_idx) % len(fallback)]


def _build_franka_visual_parts_collision_shapes(art, scene, ee_link_index):
    n = art.num_links()
    model = scene.model
    parts_by_body: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {i: [] for i in range(n)}
    art_names = list(getattr(scene.metadata, "articulation_link_names", []))

    for shape in model.shapes:
        st = shape.type.name
        bi = int(shape.body_index)
        if bi < 0 or bi >= n or st == "Plane":
            continue
        if st == "Box":
            he = np.asarray(shape.box_half_extents, dtype=np.float32).ravel()[:3]
            v, f = make_box_mesh(he)
        elif st == "Sphere":
            v, f = make_sphere_mesh(float(shape.sphere_radius), n_lat=12, n_lon=22)
        elif st == "Cylinder":
            v, f = make_cylinder_mesh(
                float(shape.cylinder_radius),
                float(shape.cylinder_half_length),
                n_segments=26,
            )
        else:
            continue
        v_body = transform_vertices(np.asarray(v, dtype=np.float32), shape.local_transform)
        parts_by_body[bi].append((np.asarray(v_body, dtype=np.float32), np.asarray(f, dtype=np.int32)))

    children = {i: [] for i in range(n)}
    for child in range(n):
        p = int(art.joints[child].parent)
        if p >= 0:
            children[p].append(child)

    for i in range(n):
        if parts_by_body[i]:
            continue
        t = i / max(1, n - 1)
        hub_r = float(np.clip(0.040 - 0.018 * t, 0.012, 0.045))
        if i == 0:
            hub_r *= 1.15
        vh, fh = make_sphere_mesh(hub_r, n_lat=10, n_lon=20)
        parts_by_body[i].append((np.asarray(vh, dtype=np.float32), np.asarray(fh, dtype=np.int32)))
        for child in children.get(i, []):
            off = np.asarray(art.joints[child].parent_to_joint.position, dtype=np.float32).ravel()[:3]
            if float(np.linalg.norm(off)) < 1e-4:
                continue
            r_seg = float(np.clip(hub_r * 0.42, 0.006, 0.018))
            vs, fs = _cylinder_between_local_points(
                np.zeros(3, dtype=np.float32), off, r_seg, n_segments=18
            )
            parts_by_body[i].append((np.asarray(vs, dtype=np.float32), np.asarray(fs, dtype=np.int32)))

    out_parts = []
    for body_idx in range(n):
        if not parts_by_body[body_idx]:
            continue
        merged_v, merged_f = _concat_mesh_parts(parts_by_body[body_idx])
        out_parts.append(
            {
                "name": f"fr3_body_visual_{body_idx}",
                "body_index": int(body_idx),
                "verts_local": merged_v.astype(np.float32, copy=False),
                "faces": merged_f.astype(np.int32, copy=False),
                "color": _fr3_component_color(
                    body_idx,
                    art_names[body_idx] if body_idx < len(art_names) else None,
                ),
            }
        )

    if int(ee_link_index) >= 0 and int(ee_link_index) < n:
        vt, ft = _cylinder_along_neg_z_from_origin(0.05, 0.0085, n_segments=16)
        out_parts.append(
            {
                "name": "fr3_tcp_marker",
                "body_index": int(ee_link_index),
                "verts_local": np.asarray(vt, dtype=np.float32),
                "faces": np.asarray(ft, dtype=np.int32),
                "color": (0.95, 0.77, 0.22),
            }
        )
    return out_parts


def build_franka_visual_parts(art, ee_link_index, scene, scene_metadata, urdf_model, urdf_path):
    """Create visual parts: mesh(default), DAE-only, or collision fallback."""
    mode = _ik_arm_visual_mode()
    if mode == "collision":
        out = _build_franka_visual_parts_collision_shapes(art, scene, ee_link_index)
        if out:
            return out
        _log_info("collision-shape visual empty; falling back to URDF visual (DAE)")

    n = art.num_links()
    children = {i: [] for i in range(n)}
    for child in range(n):
        p = int(art.joints[child].parent)
        if p >= 0:
            children[p].append(child)

    urdf_links_by_name = {link.name: link for link in getattr(urdf_model, "links", [])}
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    urdf_pkg_root_dir = os.path.dirname(urdf_dir)
    urdf_data_root_dir = os.path.dirname(urdf_pkg_root_dir)
    current_pkg_name = os.path.basename(urdf_pkg_root_dir)

    def _resolve_mesh_path(mesh_filename: str) -> str:
        mesh_filename = str(mesh_filename)
        if mesh_filename.startswith("package://"):
            rest = mesh_filename[len("package://") :].lstrip("/")
            parts = rest.split("/", 1)
            if len(parts) == 2:
                pkg_name, rel_path = parts
                candidates = [
                    os.path.join(urdf_data_root_dir, pkg_name, rel_path),
                    os.path.join(urdf_pkg_root_dir, pkg_name, rel_path),
                ]
                if pkg_name == current_pkg_name:
                    candidates.append(os.path.join(urdf_pkg_root_dir, rel_path))
                for c in candidates:
                    if os.path.exists(c):
                        return c
                return candidates[0]
            return os.path.join(urdf_data_root_dir, rest)
        if mesh_filename.startswith("file://"):
            return mesh_filename[len("file://") :]
        if os.path.isabs(mesh_filename):
            return mesh_filename
        return os.path.join(urdf_dir, mesh_filename)

    colored_parts: list[tuple[int, np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    art_names = list(getattr(scene_metadata, "articulation_link_names", []))
    for body_idx, link_name in enumerate(art_names):
        if body_idx < 0 or body_idx >= n:
            break
        urdf_link = urdf_links_by_name.get(link_name)
        if urdf_link is None:
            continue
        for visual in getattr(urdf_link, "visuals", []):
            geometry = visual.geometry
            if geometry.type == novaphy.UrdfGeometryType.Mesh:
                mesh_uri = str(geometry.mesh_filename)
                tf_origin = visual.origin
                mesh_scale_arr = np.asarray(geometry.mesh_scale, dtype=np.float32).ravel()
                if mesh_scale_arr.size == 1:
                    mesh_scale = np.full((1, 3), float(mesh_scale_arr[0]), dtype=np.float32)
                else:
                    mesh_scale = mesh_scale_arr[:3].reshape(1, 3)
                dae_path = _resolve_mesh_path(mesh_uri)
                loaded = False
                try:
                    submeshes = _load_collada_dae_submeshes(dae_path)
                    for sv, sf, scolor in submeshes:
                        verts_scaled = np.asarray(sv, dtype=np.float32) * mesh_scale
                        verts_local = transform_vertices(verts_scaled, tf_origin)
                        v_clean, f_clean = _sanitize_triangle_mesh(verts_local, sf)
                        if int(f_clean.shape[0]) <= 0:
                            continue
                        colored_parts.append(
                            (
                                body_idx,
                                v_clean.astype(np.float32, copy=False),
                                f_clean.astype(np.int32, copy=False),
                                _sanitize_rgb_color(scolor),
                            )
                        )
                    loaded = True
                except Exception as e:
                    _log_info(f"DAE submesh load failed for {dae_path}: {e}")
                if not loaded:
                    try:
                        verts_mesh, faces_mesh = _load_collada_dae_as_trimesh(dae_path)
                        verts_scaled = np.asarray(verts_mesh, dtype=np.float32) * mesh_scale
                        verts_local = transform_vertices(verts_scaled, tf_origin)
                        v_clean, f_clean = _sanitize_triangle_mesh(verts_local, faces_mesh)
                        if int(f_clean.shape[0]) <= 0:
                            continue
                        colored_parts.append(
                            (
                                body_idx,
                                v_clean.astype(np.float32, copy=False),
                                f_clean.astype(np.int32, copy=False),
                                _sanitize_rgb_color(_fr3_component_color(body_idx, link_name)),
                            )
                        )
                    except Exception as e2:
                        _log_info(f"DAE fallback also failed for {dae_path}: {e2}")
            elif geometry.type == novaphy.UrdfGeometryType.Box:
                he = np.asarray(geometry.size, dtype=np.float32).reshape(3,) * 0.5
                v, f = make_box_mesh(he)
                v_local = transform_vertices(np.asarray(v, dtype=np.float32), visual.origin)
                v_clean, f_clean = _sanitize_triangle_mesh(v_local, f)
                if int(f_clean.shape[0]) > 0:
                    colored_parts.append((body_idx, v_clean, f_clean, _sanitize_rgb_color((0.8, 0.8, 0.8))))
            elif geometry.type == novaphy.UrdfGeometryType.Sphere:
                v, f = make_sphere_mesh(float(geometry.radius), n_lat=14, n_lon=26)
                v_local = transform_vertices(np.asarray(v, dtype=np.float32), visual.origin)
                v_clean, f_clean = _sanitize_triangle_mesh(v_local, f)
                if int(f_clean.shape[0]) > 0:
                    colored_parts.append((body_idx, v_clean, f_clean, _sanitize_rgb_color((0.8, 0.8, 0.8))))
            elif geometry.type == novaphy.UrdfGeometryType.Cylinder:
                r = float(geometry.radius)
                half_l = float(geometry.length) * 0.5
                v, f = make_cylinder_mesh(r, half_l, n_segments=28)
                v_local = transform_vertices(np.asarray(v, dtype=np.float32), visual.origin)
                v_clean, f_clean = _sanitize_triangle_mesh(v_local, f)
                if int(f_clean.shape[0]) > 0:
                    colored_parts.append((body_idx, v_clean, f_clean, _sanitize_rgb_color((0.8, 0.8, 0.8))))

    colored_parts = _deduplicate_overlapping_faces_per_body(colored_parts)

    has_any_visual = {cp[0] for cp in colored_parts}
    for i in range(n):
        if i in has_any_visual:
            continue
        t = i / max(1, n - 1)
        hub_r = float(np.clip(0.040 - 0.018 * t, 0.012, 0.045))
        if i == 0:
            hub_r *= 1.15
        vh, fh = make_sphere_mesh(hub_r, n_lat=10, n_lon=20)
        v_clean_h, f_clean_h = _sanitize_triangle_mesh(vh, fh)
        if int(f_clean_h.shape[0]) > 0:
            colored_parts.append((i, v_clean_h, f_clean_h, _sanitize_rgb_color((0.7, 0.7, 0.7))))
        for child in children.get(i, []):
            off = np.asarray(art.joints[child].parent_to_joint.position, dtype=np.float32).ravel()[:3]
            if float(np.linalg.norm(off)) < 1e-4:
                continue
            r_seg = float(np.clip(hub_r * 0.42, 0.006, 0.018))
            vs, fs = _cylinder_between_local_points(
                np.zeros(3, dtype=np.float32), off, r_seg, n_segments=18
            )
            v_clean_s, f_clean_s = _sanitize_triangle_mesh(vs, fs)
            if int(f_clean_s.shape[0]) > 0:
                colored_parts.append((i, v_clean_s, f_clean_s, _sanitize_rgb_color((0.7, 0.7, 0.7))))

    out_parts = []
    part_counter: dict[int, int] = {}
    for body_idx, verts, faces, color in colored_parts:
        j = part_counter.get(body_idx, 0)
        part_counter[body_idx] = j + 1
        out_parts.append(
            {
                "name": f"fr3_body_{body_idx}_mat_{j}",
                "body_index": int(body_idx),
                "verts_local": verts.astype(np.float32, copy=False),
                "faces": faces.astype(np.int32, copy=False),
                "color": _sanitize_rgb_color(color),
            }
        )

    if int(ee_link_index) >= 0 and int(ee_link_index) < n:
        vt, ft = _cylinder_along_neg_z_from_origin(0.05, 0.0085, n_segments=16)
        out_parts.append(
            {
                "name": "fr3_tcp_marker",
                "body_index": int(ee_link_index),
                "verts_local": np.asarray(vt, dtype=np.float32),
                "faces": np.asarray(ft, dtype=np.int32),
                "color": _sanitize_rgb_color((0.95, 0.77, 0.22)),
            }
        )
    return out_parts


__all__ = [
    "build_franka_ik_setup",
    "build_franka_visual_parts",
    "find_articulation_link_index_by_name",
]
