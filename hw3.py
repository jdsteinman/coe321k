import numpy as np
import matplotlib.pyplot as plt
from truss2d import truss2d, read_inputs

## Test function
path = "./inputs3/"
nodes = path + "nodes.txt"
elements = path + "elements.txt"
forces = path + "forces.txt"
disp = path + "displacements.txt"

u, eps, stress, Fi, Fe = truss2d(nodes, elements, forces, disp, 0)
u=u.reshape((-1, 2))
print('{:^8} {:^10} {:^10}'.format("node","u (m)","v (m)"))
for i, val in enumerate(u):
    print('{:^8d} {:^ 9f} {:^ 9f}'.format(i, val[0], val[1]))

# Plots
nodes, elements, _, _ = read_inputs(nodes, elements, forces, disp, index=1) 

fig, ax = plt.subplots(1,1)
for row in elements:
    x1 = nodes[int(row[0]),0]
    y1 = nodes[int(row[0]),1]
    x2 = nodes[int(row[1]),0]
    y2 = nodes[int(row[1]),1]

    ax.plot([x1,x2],[y1,y2])

fig2, ax2 = plt.subplots(1,1)
u = u.reshape((-1,2))
nodes = nodes + u
for row in elements:
    x1 = nodes[int(row[0]),0]
    y1 = nodes[int(row[0]),1]
    x2 = nodes[int(row[1]),0]
    y2 = nodes[int(row[1]),1]

    ax2.plot([x1,x2],[y1,y2])

plt.show()
