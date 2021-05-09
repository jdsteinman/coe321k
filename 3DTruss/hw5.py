import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from truss3d import *

## Test function
def main():
    path = "./inputs/"
    nodes = path + "nodes.txt"
    elements = path + "elements.txt"
    forces = path + "forces.txt"
    disp = path + "displacements.txt"

    u = truss3d(nodes, elements, forces, disp, 1)
    u = u.reshape((-1, 3))
    print('{:^8} {:^10} {:^10} {:^10}'.format("node","u/L","v/L", "w/L"))
    for i, val in enumerate(u):
        print('{:^8d} {:^ 9f} {:^ 9f} {:^ 9f}'.format(i, val[0], val[1], val[2]))

    ax = plot_deformation(nodes, elements, u)
    plt.show()

def plot_deformation(nodes_file, elements_file, disp):
    nodes = read_nodes(nodes_file)
    elements = read_elements(elements_file, 1)
    u = disp.reshape((-1,3))
    nodes = nodes + u*0.01

    ax = plt.axes(projection='3d')
    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        x1, y1, z1 = nodes[n1,0], nodes[n1,1], nodes[n1,2]
        x2, y2, z2 = nodes[n2,0], nodes[n2,1], nodes[n2,2]

        ax.plot3D([x1,x2],[y1,y2],[z1,z2])

    return ax

if __name__=="__main__":
    main()