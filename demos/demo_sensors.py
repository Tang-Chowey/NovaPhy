"""Demo: Sensor system with IMU, contact force, and frame transform sensors.

Creates a box falling onto a ground plane with IMU and contact sensors,
prints readings each step, and shows how to use the sensor API in a
simulation loop.  Demonstrates:
  - SensorIMU with center and offset sites
  - SensorContact with real force/torque readout
  - SensorFrameTransform with relative position and velocity
"""

import numpy as np
import novaphy
from novaphy.sensors import SensorIMU, SensorContact, SensorFrameTransform


def main():
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    box_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 3.0, 0.0], dtype=np.float32)),
        density=1000.0,
    )

    builder.add_site(box_idx, label="imu_center")
    builder.add_site(box_idx,
                     novaphy.Transform.from_translation(
                         np.array([0.0, 0.5, 0.0], dtype=np.float32)),
                     "imu_top")

    model = builder.build()
    world = novaphy.World(model)
    mass = model.bodies[box_idx].mass

    imu = SensorIMU(model, sites="imu_*")
    contact = SensorContact(model, body_indices=[box_idx])
    frame = SensorFrameTransform(model,
                                 target_sites="imu_top",
                                 reference_sites="imu_center")

    print(f"IMU sensors: {imu.num_sensors} (sites matching 'imu_*')")
    print(f"Contact sensors: {contact.num_sensors}")
    print(f"Frame transform sensors: {frame.num_sensors}")
    print(f"Box mass: {mass:.1f} kg  (expected contact Fy ≈ {mass * 9.81:.1f} N at rest)")
    print()

    dt = 1.0 / 120.0
    for step in range(600):
        world.step(dt)

        imu.update(world.state, dt)
        contact.update(world.state, world.contacts, dt)
        frame.update(world.state)

        if step % 60 == 0:
            t = step * dt
            pos = world.state.transforms[box_idx].position
            print(f"t={t:.2f}s  pos_y={pos[1]:.3f}")
            print(f"  IMU[0] acc={imu.accelerometer[0]}  gyro={imu.gyroscope[0]}")
            print(f"  Contact n={contact.num_contacts[0]}  "
                  f"force={contact.forces[0]}  torque={contact.torques[0]}")
            print(f"  Frame offset={frame.positions[0]}  "
                  f"rel_vel={frame.linear_velocities[0]}")
            print()

    print("Done. Box final y:", world.state.transforms[box_idx].position[1])


if __name__ == "__main__":
    main()
