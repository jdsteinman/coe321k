from frame2d import read_inputs, solve_frame
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "./inputs/"
    nodes = path + "nodes.txt"
    elements = path + "elements.txt"
    forces = path + "forces.txt"
    disp = path + "displacements.txt"

    u, Fe = solve_frame(nodes, elements, forces, disp)

    print("\nu\n", u.reshape((-1,1)))
    print("\nFe\n", Fe.reshape((-1,1)))

    fig, ax = plt.subplots(1,1)
    nodes, elements, fbc, dbc = read_inputs(nodes, elements, forces, disp, index=0)
    ax.set_aspect('equal', adjustable='box')
    for row in elements:
        x1 = nodes[int(row[0]),0]
        y1 = nodes[int(row[0]),1]
        x2 = nodes[int(row[1]),0]
        y2 = nodes[int(row[1]),1]

        ax.plot([x1,x2],[y1,y2])

    fig2, ax2 = plt.subplots(1,1)
    ax2.set_aspect('equal', adjustable='box')
    u = u.reshape((-1,3))
    nodes = nodes + u[:,0:2]
    for row in elements:
        x1 = nodes[int(row[0]),0]
        y1 = nodes[int(row[0]),1]
        x2 = nodes[int(row[1]),0]
        y2 = nodes[int(row[1]),1]

        ax2.plot([x1,x2],[y1,y2])

    plt.show()



if __name__=="__main__":
    main()