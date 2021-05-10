import numpy as np
import matplotlib.pyplot as plt
from solid2d import *
from math import isclose

def solve6():
    path = "./inputs/"
    nodes = path + "nodes6.txt"
    elements = path + "elements6.txt"
    forces = path + "forces6.txt"
    disp = path + "displacements6.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)
    np.savetxt("output/stress6.txt", stress)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    ele_left = np.arange(123,147,4)
    ele_bottom = np.arange(0,24,4)

    f1, a1 = plot_deformation(nodes, elements, u)
    # f2, a2 = plot_stress_left(nodes, elements, stress)
    # f3, a3 = plot_stress_bottom(nodes, elements, stress)
    plt.show()

def solve12():
    path = "./inputs/"
    nodes = path + "nodes12.txt"
    elements = path + "elements12.txt"
    forces = path + "forces12.txt"
    disp = path + "displacements12.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)
    np.savetxt("output/stress12.txt", stress)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    ele_left = np.arange(531,579,4)
    ele_bottom = np.arange(0,48,4)

    f1, a1 = plot_deformation(nodes, elements, u)
    # f2, a2 = plot_stress_left(nodes, elements, stress)
    # f3, a3 = plot_stress_bottom(nodes, elements, stress)
    plt.show()

def solve24():
    path = "./inputs/"
    nodes = path + "nodes24.txt"
    elements = path + "elements24.txt"
    forces = path + "forces24.txt"
    disp = path + "displacements24.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)
    np.savetxt("output/stress24.txt", stress)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    ele_left = np.arange(2211,2307,4)
    ele_bottom = np.arange(0,96,4)

    f1, a1 = plot_deformation(nodes, elements, u)
    # f2, a2 = plot_stress_left(nodes, elements, stress)
    # f3, a3 = plot_stress_bottom(nodes, elements, stress)
    plt.show()

def solveR():
    path = "./inputs/"
    nodes = path + "nodesR.txt"
    elements = path + "elementsR.txt"
    forces = path + "forcesR.txt"
    disp = path + "displacementsR.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)
    np.savetxt("output/stressR.txt", stress)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    ele_left = np.arange(531,579,4)
    ele_bottom = np.arange(0,48,4)

    f1, a1 = plot_deformation(nodes, elements, u)
    f2, a2 = plot_stress_left(nodes, elements, stress)
    f3, a3 = plot_stress_bottom(nodes, elements, stress
    plt.show()


def plot_deformation(nodes_file, elements_file, disp):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file, 1)

    u = disp.reshape((-1,2))
    u = u*0.1

    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')

    nodes_d = nodes + u  # deformed nodes
    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

        ax.plot([x1,x2],[y1,y2], c='#bebebe', zorder=1)
        ax.plot([x1,x3],[y1,y3], c='#bebebe', zorder=1)
        ax.plot([x2,x3],[y2,y3], c='#bebebe', zorder=1)

        # Deformed nodes
        X1, Y1 = nodes_d[n1,0], nodes_d[n1,1]
        X2, Y2 = nodes_d[n2,0], nodes_d[n2,1]
        X3, Y3 = nodes_d[n3,0], nodes_d[n3,1]

        ax.plot([X1,X2],[Y1,Y2], c='r', zorder=2)
        ax.plot([X1,X3],[Y1,Y3], c='r', zorder=2)
        ax.plot([X2,X3],[Y2,Y3], c='r', zorder=2)

        xbar = (x1+x2+x3)/3
        ybar = (y1+y2+y3)/3

    ax.set_title(r"Deformation of plate, scaled by 0.1E/R$\sigma$")
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")

    legend_elements = [plt.Line2D([0], [0], color='r', lw=2, label='Deformed'),
                    plt.Line2D([0], [0], color='#bebebe', lw=2, label='Undeformed')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), ncol=1)
    return fig, ax

def plot_stress_left(nodes_file, elements_file, stress):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file, 1)
    
    y = np.array([])
    sigma_xx = np.array([])
    sigma_yy = np.array([])
    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

        # skip if not two nodes on left boundary
        a = isclose(x1,0.0)
        b = isclose(x2,0.0)
        c = isclose(x3,0.0)
        if not ((a and b) or (a and c) or (b and c)): continue

        xbar = (x1+x2+x3)/3
        ybar = (y1+y2+y3)/3

        y = np.append(y, ybar)
        sigma_xx = np.append(sigma_xx, stress[e,0])
        sigma_yy = np.append(sigma_yy, stress[e,1])

    # sort
    ind = np.argsort(y)
    y = y[ind]
    sigma_xx = sigma_xx[ind]
    sigma_yy = sigma_yy[ind]

    # Extrapolate 
    sigma_xx_0 = sigma_xx[0] - (y[0]-1)*(sigma_xx[1]-sigma_xx[0])/(y[1]-y[0])
    sigma_yy_0 = sigma_yy[0] - (y[0]-1)*(sigma_yy[1]-sigma_yy[0])/(y[1]-y[0])

    fig, ax = plt.subplots(1,1)
    ax.scatter(y,sigma_xx, c='r', label="sigma_xx")
    ax.scatter(1,sigma_xx_0, edgecolors='r', facecolors='none')
    ax.scatter(y,sigma_yy, c='b', label="sigma_yy")
    ax.scatter(1,sigma_yy_0, edgecolors='b', facecolors='none')
    ax.set_xlabel("y/r")
    ax.set_ylabel(r"$\sigma/\sigma_{applied}$")
    ax.set_title("Stress along left wall")
    ax.legend()

    print("sigma_xx at (0,1): %8.6f"%sigma_xx_0)
    print("sigma_yy at (0,1): %8.6f"%sigma_yy_0)

    return fig, ax

def plot_stress_bottom(nodes_file, elements_file, stress):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file, 1)
    
    i = np.arange(0,48,4)
    x = np.array([])
    sigma_xx = np.array([])
    sigma_yy = np.array([])
    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

        # skip if not two nodes on bottom boundary
        a = isclose(y1,0.0)
        b = isclose(y2,0.0)
        c = isclose(y3,0.0)
        if not ((a and b) or (a and c) or (b and c)): continue

        xbar = (x1+x2+x3)/3
        ybar = (y1+y2+y3)/3

        x = np.append(x, xbar)
        sigma_xx = np.append(sigma_xx, stress[e,0])
        sigma_yy = np.append(sigma_yy, stress[e,1])

    # sort
    ind = np.argsort(x)
    x = x[ind]
    sigma_xx = sigma_xx[ind]
    sigma_yy = sigma_yy[ind]

    # Extrapolate 
    sigma_xx_0 = sigma_xx[0] - (x[0]-1)*(sigma_xx[1]-sigma_xx[0])/(x[1]-x[0])
    sigma_yy_0 = sigma_yy[0] - (x[0]-1)*(sigma_yy[1]-sigma_yy[0])/(x[1]-x[0])

    fig, ax = plt.subplots(1,1)
    ax.scatter(x,sigma_xx, c='r', label="sigma_xx")
    ax.scatter(x,sigma_yy, c='b', label="sigma_yy")
    ax.scatter(1,sigma_xx_0, edgecolors='r', facecolors='none')
    ax.scatter(1,sigma_yy_0, edgecolors='b', facecolors='none')

    ax.set_xlabel("x/r")
    ax.set_ylabel(r"$\sigma/\sigma_{applied}$")
    ax.set_title("Stress along bottom wall")
    ax.legend()

    print("sigma_xx at (1,0): %8.6f"%sigma_xx_0)
    print("sigma_yy at (1,0): %8.6f"%sigma_yy_0)

    return fig, ax

if __name__=="__main__":
    solveR()