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

if __name__=="__main__":
    main()