"""Step 7: FluidMaterial multi-material, coupling, vorticity defaults (TDD)."""

import numpy as np
import numpy.testing as npt
import novaphy


def test_fluid_material_defaults():
    m = novaphy.FluidMaterial()
    npt.assert_allclose(m.rest_density, 1000.0, atol=1e-3)


def test_fluid_block_def_has_material_index_default_zero():
    b = novaphy.FluidBlockDef()
    assert b.material_index == 0


def test_pbf_settings_default_vorticity_epsilon_is_mild():
    """Enhanced default so thin vortices get weak confinement when enabled."""
    s = novaphy.PBFSettings()
    assert s.vorticity_epsilon > 0.0


def test_two_materials_different_particle_masses():
    """Two blocks with different FluidMaterial rest_density -> different particle masses."""
    model = novaphy.ModelBuilder().build()
    model.fluid_materials = [novaphy.FluidMaterial(), novaphy.FluidMaterial()]
    model.fluid_materials[0].rest_density = 1000.0
    model.fluid_materials[1].rest_density = 500.0

    spacing = 0.05
    b0 = novaphy.FluidBlockDef()
    b0.lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b0.upper = np.array([0.1, 0.05, 0.05], dtype=np.float32)
    b0.particle_spacing = spacing
    b0.material_index = 0

    b1 = novaphy.FluidBlockDef()
    b1.lower = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    b1.upper = np.array([2.1, 0.05, 0.05], dtype=np.float32)
    b1.particle_spacing = spacing
    b1.material_index = 1

    model.fluid_blocks = [b0, b1]

    pbf = novaphy.PBFSettings()
    pbf.kernel_radius = spacing * 4.0
    world = novaphy.World(model, novaphy.SolverSettings(), novaphy.XPBDSolverSettings(), pbf)

    masses = list(world.state.fluid_state.particle_masses)
    assert len(masses) >= 2
    m_heavy = max(masses)
    m_light = min(masses)
    npt.assert_allclose(m_heavy / m_light, 2.0, rtol=0.05)


def test_unified_world_articulation_plus_fluid_steps():
    """Articulation boundary + fluid in one World should step without error."""
    from test_articulated_world import _build_single_hip_scene

    scene = _build_single_hip_scene(root_height=0.7)
    model = scene.model
    model.articulations = [scene.articulation]

    block = novaphy.FluidBlockDef()
    block.lower = np.array([-0.3, 0.05, -0.3], dtype=np.float32)
    block.upper = np.array([0.3, 0.35, 0.3], dtype=np.float32)
    block.particle_spacing = 0.08
    model.fluid_blocks = [block]

    pbf = novaphy.PBFSettings()
    pbf.kernel_radius = 0.2
    pbf.solver_iterations = 2
    pbf.vorticity_epsilon = 0.0  # keep test deterministic

    world = novaphy.World(
        model,
        novaphy.SolverSettings(),
        novaphy.XPBDSolverSettings(),
        pbf,
        fluid_boundary_extent=0.5,
    )

    assert world.state.fluid_state.num_particles > 0
    for _ in range(5):
        world.step(1.0 / 60.0)

    assert np.isfinite(world.state.fluid_state.positions[0][0])


def test_pbf_vorticity_enabled_runs_without_nan():
    state, settings, mass = _make_tiny_pbf_state()
    settings.vorticity_epsilon = 0.05
    solver = novaphy.PBFSolver(settings)
    g = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(3):
        solver.step(state, 1.0 / 60.0, g, mass)
    for p in state.positions:
        assert np.all(np.isfinite(p))
    for v in state.velocities:
        assert np.all(np.isfinite(v))


def _make_tiny_pbf_state():
    block = novaphy.FluidBlockDef()
    block.lower = np.array([0, 0, 0], dtype=np.float32)
    block.upper = np.array([0.15, 0.15, 0.15], dtype=np.float32)
    block.particle_spacing = 0.05
    positions = novaphy.generate_fluid_block(block)
    settings = novaphy.PBFSettings()
    settings.kernel_radius = 0.12
    settings.rest_density = 1000.0
    settings.solver_iterations = 3
    state = novaphy.ParticleState()
    state.init(positions)
    mass = settings.particle_mass(block.particle_spacing)
    return state, settings, mass
