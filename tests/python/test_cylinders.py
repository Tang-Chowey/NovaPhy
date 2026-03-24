import numpy as np
import novaphy

def test_cylinder_plane_collision():
    builder = novaphy.ModelBuilder()
    
    # Ground plane
    ground_shape = novaphy.CollisionShape.make_plane(
        normal=[0.0, 1.0, 0.0], offset=0.0
    )
    builder.add_shape(ground_shape)
    
    # Cylinder radius=0.5, length=2.0 (half_length=1.0)
    cyl_body = novaphy.RigidBody.from_cylinder(mass=1.0, radius=0.5, length=2.0)
    
    # Quat for 90 deg around X (so local Z is aligned with world Y)
    qx, qy, qz, qw = np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)
    t = novaphy.Transform(
        position=[0.0, 5.0, 0.0],
        rotation=[qx, qy, qz, qw]
    )
    
    body_idx = builder.add_body(cyl_body, t)
    
    cyl_shape = novaphy.CollisionShape.make_cylinder(
        radius=0.5, half_length=1.0, body_idx=body_idx
    )
    builder.add_shape(cyl_shape)
    
    model = builder.build()
    world = novaphy.World(model)
    
    dt = 1.0 / 120.0
    for _ in range(240):
        world.step(dt)
        
    pos = world.state.transforms[body_idx].position
    # Center should rest at Y = 1.0 (half_length)
    assert pos[1] >= 0.95, f"Cylinder too low: y={pos[1]}"
    assert pos[1] <= 1.05, f"Cylinder too high/bouncing: y={pos[1]}"

def test_cylinder_sphere_collision():
    builder = novaphy.ModelBuilder()
    
    cyl_body = novaphy.RigidBody.make_static()
    t1 = novaphy.Transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    b1 = builder.add_body(cyl_body, t1)
    builder.add_shape(novaphy.CollisionShape.make_cylinder(0.5, 1.0, b1))
    
    sph_body = novaphy.RigidBody.from_sphere(mass=1.0, radius=0.25)
    t2 = novaphy.Transform([0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    b2 = builder.add_body(sph_body, t2)
    builder.add_shape(novaphy.CollisionShape.make_sphere(0.25, b2))
    
    model = builder.build()
    world = novaphy.World(model)
    
    # Sphere falls onto static cylinder (cylinder local Z is world Z)
    # Give it enough time to settle
    for _ in range(240):
        world.step(1.0/120.0)
        
    pos = world.state.transforms[b2].position
    # Center should rest at Y = cylinder radius (0.5) + sphere radius (0.25) = 0.75
    assert pos[1] >= 0.70, f"Sphere too low: y={pos[1]}"
    assert pos[1] <= 0.80, f"Sphere too high: y={pos[1]}"

