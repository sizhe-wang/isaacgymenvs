import pybullet as p

p.connect(p.GUI)
p.loadURDF('./yumi.urdf', useFixedBase=True)

for _ in range(10000000):
    p.stepSimulation()
