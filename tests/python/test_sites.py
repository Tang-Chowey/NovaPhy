"""Tests for NovaPhy Site infrastructure."""

import numpy as np
import numpy.testing as npt
import novaphy


def test_site_default_construction():
    site = novaphy.Site()
    assert site.label == ""
    assert site.body_index == -1
    assert site.articulation_index == -1
    assert site.link_index == -1
    assert not site.is_articulation_site()


def test_site_attributes():
    site = novaphy.Site()
    site.label = "imu_0"
    site.body_index = 2
    tf = novaphy.Transform.from_translation(
        np.array([0.0, 0.1, 0.0], dtype=np.float32))
    site.local_transform = tf
    assert site.label == "imu_0"
    assert site.body_index == 2
    npt.assert_allclose(site.local_transform.position, [0.0, 0.1, 0.0], atol=1e-6)


def test_model_builder_add_site():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(half)
    site_idx = builder.add_site(body_idx, label="sensor_0")
    assert site_idx == 0
    assert builder.num_sites == 1


def test_model_stores_sites():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b0 = builder.add_shape_box(half)
    b1 = builder.add_shape_box(half)
    builder.add_site(b0, label="site_a")
    builder.add_site(b1, label="site_b")
    model = builder.build()
    assert model.num_sites == 2
    assert model.sites[0].label == "site_a"
    assert model.sites[0].body_index == b0
    assert model.sites[1].label == "site_b"
    assert model.sites[1].body_index == b1
