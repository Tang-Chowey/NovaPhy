from pathlib import Path

import numpy as np

import novaphy


QUADRUPED_URDF_PATH = Path("demos/data/quadruped.urdf")
BUILD_ASSET_DIR = Path("build/test_assets")

ADJACENT_CHAIN_URDF = """<robot name=\"adjacent_chain\">
  <link name=\"base\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.2\" ixy=\"0\" ixz=\"0\" iyy=\"0.2\" iyz=\"0\" izz=\"0.2\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"1 1 1\"/></geometry>
    </collision>
  </link>
  <link name=\"mid\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.2\" ixy=\"0\" ixz=\"0\" iyy=\"0.2\" iyz=\"0\" izz=\"0.2\"/>
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
      <inertia ixx=\"0.2\" ixy=\"0\" ixz=\"0\" iyy=\"0.2\" iyz=\"0\" izz=\"0.2\"/>
    </inertial>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"1 1 1\"/></geometry>
    </collision>
  </link>
  <joint name=\"shoulder\" type=\"revolute\">
    <parent link=\"base\"/>
    <child link=\"mid\"/>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"10\" velocity=\"2\"/>
  </joint>
  <joint name=\"elbow\" type=\"revolute\">
    <parent link=\"mid\"/>
    <child link=\"tip\"/>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"10\" velocity=\"2\"/>
  </joint>
</robot>
"""

FIXED_CLUSTER_URDF = """<robot name=\"fixed_cluster\">
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
      <origin xyz=\"0 0.1 0\" rpy=\"0 0 0\"/>
      <geometry><sphere radius=\"0.2\"/></geometry>
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
      <geometry><box size=\"0.5 0.5 0.5\"/></geometry>
    </collision>
  </link>
  <joint name=\"mount\" type=\"fixed\">
    <parent link=\"base\"/>
    <child link=\"sensor\"/>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
  </joint>
  <joint name=\"hinge\" type=\"revolute\">
    <parent link=\"sensor\"/>
    <child link=\"tip\"/>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"10\" velocity=\"2\"/>
  </joint>
</robot>
"""


def _write_asset(name: str, payload: str) -> Path:
    BUILD_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_ASSET_DIR / name
    path.write_text(payload, encoding="utf-8")
    return path


def _frozenset_pairs(pairs):
    return {frozenset(pair) for pair in pairs}


def test_quadruped_self_collision_filter_disables_all_robot_pairs():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    model = parser.parse_file(QUADRUPED_URDF_PATH)

    options = novaphy.UrdfImportOptions()
    options.enable_self_collisions = False
    result = builder.build_from_urdf(model, options)

    assert len(result.model.collision_filter_pairs) == 78
    assert len(result.metadata.filtered_link_pairs) == 78
    filtered_pairs = _frozenset_pairs(result.metadata.filtered_link_pairs)
    assert frozenset({"base", "LF_HAA"}) in filtered_pairs
    assert frozenset({"LF_HAA", "LF_THIGH"}) in filtered_pairs
    assert frozenset({"LF_SHANK", "RH_SHANK"}) in filtered_pairs


def test_same_cluster_and_parent_child_filters_are_generated_when_collapse_is_enabled():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    model = parser.parse_file(_write_asset("fixed_cluster_filter.urdf", FIXED_CLUSTER_URDF))

    options = novaphy.UrdfImportOptions()
    options.collapse_fixed_joints = True
    options.enable_self_collisions = True
    result = builder.build_from_urdf(model, options)

    assert len(result.model.collision_filter_pairs) == 3
    filtered_pairs = _frozenset_pairs(result.metadata.filtered_link_pairs)
    assert frozenset({"base", "sensor"}) in filtered_pairs
    assert frozenset({"base", "tip"}) in filtered_pairs
    assert frozenset({"sensor", "tip"}) in filtered_pairs


def test_world_skips_filtered_parent_child_pairs_but_keeps_non_adjacent_contacts():
    parser = novaphy.UrdfParser()
    builder = novaphy.SceneBuilderEngine()
    model = parser.parse_file(_write_asset("adjacent_chain_filter.urdf", ADJACENT_CHAIN_URDF))

    options = novaphy.UrdfImportOptions()
    options.enable_self_collisions = True
    result = builder.build_from_urdf(model, options)

    filtered_pairs = _frozenset_pairs(result.metadata.filtered_link_pairs)
    assert frozenset({"base", "mid"}) in filtered_pairs
    assert frozenset({"mid", "tip"}) in filtered_pairs
    assert frozenset({"base", "tip"}) not in filtered_pairs

    world = novaphy.World(result.model)
    world.set_gravity(np.zeros(3, dtype=np.float32))
    world.step(1.0 / 120.0)

    contact_pairs = {frozenset({int(c.body_a), int(c.body_b)}) for c in world.contacts}
    assert frozenset({0, 2}) in contact_pairs
    assert frozenset({0, 1}) not in contact_pairs
    assert frozenset({1, 2}) not in contact_pairs
