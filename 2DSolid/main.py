import numpy as np
from solid2d import *

def solve6():
    path = "./inputs/"
    nodes = path + "nodes6.txt"
    elements = path + "elements6.txt"
    forces = path + "forces6.txt"
    disp = path + "displacements6.txt"

    u, Fe = solve_solid(nodes, elements, forces, disp, index=1)

    print("\nu\n", u.reshape((-1,1)))
    print("\nFe\n", Fe.reshape((-1,1)))

def solve12():
    path = "./inputs/"
    nodes = path + "nodes12.txt"
    elements = path + "elements12.txt"
    forces = path + "forces12.txt"
    disp = path + "displacements12.txt"

    u, Fe = solve_solid(nodes, elements, forces, disp, index=1)

    print("\nu\n", u.reshape((-1,1)))
    print("\nFe\n", Fe.reshape((-1,1)))

def solve24():
    path = "./inputs/"
    nodes = path + "nodes24.txt"
    elements = path + "elements24.txt"
    forces = path + "forces24.txt"
    disp = path + "displacements24.txt"

    u, Fe = solve_solid(nodes, elements, forces, disp, index=1)

    print("\nu\n", u.reshape((-1,1)))
    print("\nFe\n", Fe.reshape((-1,1)))

if __name__=="__main__":
    solve6()