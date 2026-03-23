from pathlib import Path

import novaphy


QUADRUPED_URDF_PATH = Path("demos/data/quadruped.urdf")


def test_newton_quadruped_asset_matches_expected_structure():
    parser = novaphy.UrdfParser()
    model = parser.parse_file(QUADRUPED_URDF_PATH)

    assert model.name == "quadruped"
    assert len(model.links) == 13
    assert len(model.joints) == 12
    assert all(len(link.visuals) == 0 for link in model.links)
    assert all(len(link.collisions) == 1 for link in model.links)
    assert all(
        collision.geometry.type == novaphy.UrdfGeometryType.Cylinder
        for link in model.links
        for collision in link.collisions
    )
    assert all(joint.type == "revolute" for joint in model.joints)
    assert all(joint.parent_link for joint in model.joints)
    assert all(joint.child_link for joint in model.joints)


def test_newton_quadruped_joint_name_order_matches_demo_standing_pose_suffix():
    parser = novaphy.UrdfParser()
    model = parser.parse_file(QUADRUPED_URDF_PATH)

    assert [joint.name for joint in model.joints] == [
        "LF_HAA",
        "LF_HFE",
        "LF_KFE",
        "RF_HAA",
        "RF_HFE",
        "RF_KFE",
        "LH_HAA",
        "LH_HFE",
        "LH_KFE",
        "RH_HAA",
        "RH_HFE",
        "RH_KFE",
    ]
