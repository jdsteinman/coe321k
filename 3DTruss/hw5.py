import numpy as np
import matplotlib.pyplot as plt
from truss3d import truss3d, read_inputs

## Test function
path = "./hw5/inputs/"
nodes = path + "nodes.txt"
elements = path + "elements.txt"
forces = path + "forces.txt"
disp = path + "displacements.txt"

u = truss3d(nodes, elements, forces, disp, 1)
u = u.reshape((-1, 3))
print('{:^8} {:^10} {:^10} {:^10}'.format("node","u/L","v/L", "w/L"))
for i, val in enumerate(u):
    print('{:^8d} {:^ 9f} {:^ 9f} {:^ 9f}'.format(i, val[0], val[1], val[2]))
