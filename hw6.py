from frame2d import read_inputs, solve_frame
import numpy as np

def main():
    path = "./hw6/inputs/"
    nodes = path + "nodes.txt"
    elements = path + "elements.txt"
    forces = path + "forces.txt"
    disp = path + "displacements.txt"

    u, eps, stress, Fi, Fe = solve_frame(nodes, elements, forces, disp)

    print("\nu\n", u.reshape((-1,1)))
    print("\neps\n", eps.reshape((-1,1)))
    print("\nsigma\n", stress.reshape((-1,1)))
    print("\nFi\n", Fi.reshape((-1,1)))
    print("\nFe\n", Fe.reshape((-1,1)))

if __name__=="__main__":
    main()