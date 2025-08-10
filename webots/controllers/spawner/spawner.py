from controller import Supervisor
import random
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

N_CARS = 15
Z_FIXED = 0.31

XMIN, XMAX = -180.0, 180.0
YMIN, YMAX = -180.0, 180.0
AVOID_RADIUS = 3.0

root_children = supervisor.getRoot().getField("children")
spawned_positions = []


def far_enough(x, z, positions, r=AVOID_RADIUS):
    rr = r * r
    for px, pz in positions:
        if (x - px) ** 2 + (z - pz) ** 2 < rr:
            return False
    return True


def spawn_tesla(name: str):
    for _ in range(200):
        x = random.uniform(XMIN, XMAX)
        y = random.uniform(YMIN, YMAX)
        if far_enough(x, y, spawned_positions):
            break
    else:
        x = random.uniform(XMIN, XMAX)
        y = random.uniform(YMIN, YMAX)

    node_str = f"""
        TeslaModel3 {{
        translation {x:.5f} {y:.5f} {Z_FIXED:.5f}
        rotation 0 1 0 0
    }}"""
    root_children.importMFNodeFromString(-1, node_str)
    spawned_positions.append((x, y))


for i in range(N_CARS):
    spawn_tesla(f"TESLA_{i}")

while supervisor.step(timestep) != -1:
    pass
