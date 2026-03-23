from pathlib import Path

import numpy as np
import numpy.testing as npt

import novaphy


QUADRUPED_URDF_PATH = Path("demos/data/quadruped.urdf")
BUILD_ASSET_DIR = Path("build/test_assets")

VISUAL_ONLY_URDF = """<robot name=\"visual_only\">
  <link name=\"base\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\"/>
    </inertial>
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"1 1 1\"/></geometry>
    </visual>
  </link>
</robot>
"""

DYNAMIC_URDF = """<robot name=\"dynamic_joint\">
  <link name=\"base\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"1 1 1\"/></geometry>
    </collision>
  </link>
  <link name=\"tip\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"1 1 1\"/></geometry>
    </collision>
  </link>
  <joint name=\"hinge\" type=\"revolute\">
    <parent link=\"base\"/>
    <child link=\"tip\"/>
    <origin xyz=\"0 1 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.0\" upper=\"1.0\" effort=\"9.0\" velocity=\"3.0\"/>
    <dynamics damping=\"0.25\" friction=\"0.5\"/>
  </joint>
</robot>
"""

FIXED_CHAIN_URDF = """<robot name=\"fixed_chain\">
  <link name=\"base\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.2\" ixy=\"0\" ixz=\"0\" iyy=\"0.2\" iyz=\"0\" izz=\"0.2\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"0.5 0.5 0.5\"/></geometry>
    </collision>
  </link>
  <link name=\"sensor\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"0.5\"/>
      <inertia ixx=\"0.05\" ixy=\"0\" ixz=\"0\" iyy=\"0.05\" iyz=\"0\" izz=\"0.05\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 0.2\" rpy=\"0 0 0\"/>
      <geometry><sphere radius=\"0.1\"/></geometry>
    </collision>
  </link>
  <link name=\"tip\">
    <inertial>
      <origin xyz=\"0 0 -0.2\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 -0.2\" rpy=\"0 0 0\"/>
      <geometry><box size=\"0.2 0.2 0.4\"/></geometry>
    </collision>
  </link>
  <joint name=\"mount\" type=\"fixed\">
    <parent link=\"base\"/>
    <child link=\"sensor\"/>
    <origin xyz=\"0 0.3 0\" rpy=\"0 0 0\"/>
  </joint>
  <joint name=\"hinge\" type=\"revolute\">
    <parent link=\"sensor\"/>
    <child link=\"tip\"/>
    <origin xyz=\"0 0.6 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"10\" velocity=\"2\"/>
  </joint>
</robot>
"""

GEOMETRY_INERTIA_URDF = """<robot name=\"geometry_inertia\">
  <link name=\"base\">
    <inertial>
      <origin xyz=\"1 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"4.0\"/>
      <inertia ixx=\"9.0\" ixy=\"0\" ixz=\"0\" iyy=\"9.0\" iyz=\"0\" izz=\"9.0\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 -0.2\" rpy=\"0 0 0\"/>
      <geometry><cylinder radius=\"0.5\" length=\"2.0\"/></geometry>
    </collision>
  </link>
</robot>
"""


def _write_asset(name: str, payload: str) -> Path:
    BUILD_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_ASSET_DIR / name
    path.write_text(payload, encoding="utf-8")
    return path


def test_urdf_build_metadata_tracks_runtime_mappings():
    parser = novaphy.UrdfParser()
    model = parser.parse_file(QUADRUPED_URDF_PATH)
    builder = novaphy.SceneBuilderEngine()

    options = novaphy.UrdfImportOptions()
    options.enable_self_collisions = False
    result = builder.build_from_urdf(model, options)
    metadata = result.metadata

    assert metadata.import_options.enable_self_collisions is False
    assert metadata.root_link_name == "base"
    assert metadata.root_body_index == 0
    assert metadata.root_articulation_index == 0
    assert metadata.link_names[0] == "base"
    assert metadata.articulation_link_names[0] == "base"
    assert metadata.joint_names[:3] == ["LF_HAA", "LF_HFE", "LF_KFE"]
    assert metadata.dof_joint_names[:6] == ["__root__"] * 6
    assert len(metadata.dof_joint_names) == result.articulation.total_qd()
    assert metadata.dof_qd_indices == list(range(result.articulation.total_qd()))
    assert result.initial_q.shape[0] == result.articulation.total_q()
    assert result.initial_qd.shape[0] == result.articulation.total_qd()

    joint_entries = {entry.joint_name: entry for entry in metadata.joints}
    assert joint_entries["LF_HAA"].articulation_index == 1
    assert joint_entries["LF_HAA"].qd_start == 6
    assert joint_entries["LF_HFE"].parent_link_name == "LF_HAA"
    assert joint_entries["LF_HFE"].child_link_name == "LF_THIGH"
    assert joint_entries["LF_HFE"].damping == 0.0
    assert joint_entries["LF_HFE"].friction == 0.0

    link_entries = {entry.link_name: entry for entry in metadata.links}
    assert link_entries["LF_THIGH"].body_link_name == "LF_THIGH"
    assert link_entries["LF_THIGH"].collapsed is False

    shape_entries = {entry.link_name: entry for entry in metadata.shapes}
    assert shape_entries["base"].body_index == 0
    assert shape_entries["base"].body_link_name == "base"
    assert shape_entries["base"].source == "collision"
    assert all(entry.source == "collision" for entry in metadata.shapes)


def test_visual_collision_fallback_is_opt_in():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    path = _write_asset("visual_only.urdf", VISUAL_ONLY_URDF)
    model = parser.parse_file(path)

    default_result = builder.build_from_urdf(model)
    assert default_result.model.num_shapes == 0
    assert any("fallback collision is disabled" in warning for warning in default_result.warnings)

    options = novaphy.UrdfImportOptions()
    options.use_visual_collision_fallback = True
    fallback_result = builder.build_from_urdf(model, options)
    assert fallback_result.model.num_shapes == 1
    assert fallback_result.metadata.shapes[0].source == "visual_fallback"


def test_joint_dynamics_are_parsed_written_and_exposed_in_metadata():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    path = _write_asset("dynamic_joint.urdf", DYNAMIC_URDF)
    model = parser.parse_file(path)

    assert model.joints[0].damping == 0.25
    assert model.joints[0].friction == 0.5

    result = builder.build_from_urdf(model)
    assert result.metadata.joints[0].damping == 0.25
    assert result.metadata.joints[0].friction == 0.5
    written = parser.write_string(model)
    assert '<dynamics damping="0.25" friction="0.5"/>' in written


def test_root_transform_and_initial_state_are_seeded_for_floating_base():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    model = parser.parse_file(QUADRUPED_URDF_PATH)

    options = novaphy.UrdfImportOptions()
    options.root_transform = novaphy.Transform.from_translation(np.array([0.0, 0.0, 0.7], dtype=np.float32))
    result = builder.build_from_urdf(model, options)

    npt.assert_allclose(result.initial_q[:3], [0.0, 0.0, 0.7], atol=1e-6)
    npt.assert_allclose(result.initial_q[3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    npt.assert_allclose(result.initial_qd, np.zeros_like(result.initial_qd), atol=1e-6)
    npt.assert_allclose(result.model.initial_transforms[0].position, [0.0, 0.0, 0.7], atol=1e-6)


def test_fixed_joint_collapse_preserves_metadata_and_merges_shapes():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    path = _write_asset("fixed_chain.urdf", FIXED_CHAIN_URDF)
    model = parser.parse_file(path)

    options = novaphy.UrdfImportOptions()
    options.collapse_fixed_joints = True
    result = builder.build_from_urdf(model, options)

    assert result.model.num_bodies == 2
    assert result.articulation.num_links() == 2
    assert result.metadata.articulation_link_names == ["base", "tip"]

    links = {entry.link_name: entry for entry in result.metadata.links}
    assert links["sensor"].collapsed is True
    assert links["sensor"].body_link_name == "base"
    assert links["sensor"].body_index == links["base"].body_index

    joints = {entry.joint_name: entry for entry in result.metadata.joints}
    assert joints["mount"].collapsed is True
    assert joints["mount"].articulation_index == -1
    assert joints["hinge"].collapsed is False
    assert result.articulation.joints[1].parent == 0

    shape_bodies = {entry.link_name: entry.body_link_name for entry in result.metadata.shapes}
    assert shape_bodies["base"] == "base"
    assert shape_bodies["sensor"] == "base"
    assert shape_bodies["tip"] == "tip"
    assert np.isclose(result.model.bodies[0].mass, 1.5)


def test_ignore_inertial_definitions_uses_collision_geometry_for_inertia():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    path = _write_asset("geometry_inertia.urdf", GEOMETRY_INERTIA_URDF)
    model = parser.parse_file(path)

    default_result = builder.build_from_urdf(model)
    options = novaphy.UrdfImportOptions()
    options.ignore_inertial_definitions = True
    geometry_result = builder.build_from_urdf(model, options)

    default_body = default_result.model.bodies[0]
    geometry_body = geometry_result.model.bodies[0]

    npt.assert_allclose(default_body.com, [1.0, 0.0, 0.0], atol=1e-6)
    npt.assert_allclose(np.diag(default_body.inertia), [9.0, 9.0, 9.0], atol=1e-6)

    npt.assert_allclose(geometry_body.com, [0.0, 0.0, -0.2], atol=1e-6)
    npt.assert_allclose(
        np.diag(geometry_body.inertia),
        [1.5833334, 1.5833334, 0.5],
        atol=1e-4,
    )
