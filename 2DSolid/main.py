import numpy as np
import matplotlib.pyplot as plt
from solid2d import *

# plt.style.use('ggplot')

def solve6():
    path = "./inputs/"
    nodes = path + "nodes6.txt"
    elements = path + "elements6.txt"
    forces = path + "forces6.txt"
    disp = path + "displacements6.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    plot_deformation(nodes, elements, u, 1)

def solve12():
    path = "./inputs/"
    nodes = path + "nodes12.txt"
    elements = path + "elements12.txt"
    forces = path + "forces12.txt"
    disp = path + "displacements12.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    plot_deformation(nodes, elements, u, 1)

def solve24():
    path = "./inputs/"
    nodes = path + "nodes24.txt"
    elements = path + "elements24.txt"
    forces = path + "forces24.txt"
    disp = path + "displacements24.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    plot_deformation(nodes, elements, u, 1)

def solveR():
    path = "./inputs/"
    nodes = path + "nodesR.txt"
    elements = path + "elementsR.txt"
    forces = path + "forcesR.txt"
    disp = path + "displacementsR.txt"

    u, strain, stress = solve_solid(nodes, elements, forces, disp, index=1)

    # print("\nu\n", u.reshape((-1,1)))
    # print("\nstrain\n", strain)
    # print("\nstress\n", stress)

    ele_bottom = np.arange(0,48,4)
    ele_top = np.array([574, 526, 478, 430, 382, 334])

    f1, a1 = plot_deformation(nodes, elements, u, 1)
    f2, a2 = plot_stress_top(nodes, elements, stress, ele_top, 1)
    f3, a3 = plot_stress_bottom(nodes, elements, stress, ele_bottom, 1)
    plt.show()


def plot_deformation(nodes_file, elements_file, disp, index=0):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file)

    if index==1:
        elements[:,0:3]-=1

    u = disp.reshape((-1,2))
    u = u*0.1

    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal', adjustable='box')

    nodes = nodes + u
    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

        ax.plot([x1,x2],[y1,y2])
        ax.plot([x1,x3],[y1,y3])
        ax.plot([x2,x3],[y2,y3])

        xbar = (x1+x2+x3)/3
        ybar = (y1+y2+y3)/3
        # ax.text(xbar, ybar, str(e))

    ax.set_title("Deformation of plate, scaled by 0.1")
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    return fig, ax

def plot_stress_top(nodes_file, elements_file, stress, ele_list, index=0):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file)

    if index==1:
        elements[:,0:3]-=1
    
    i = np.arange(0,48,4)
    x = np.array([])
    sigma_xx = np.array([])
    sigma_yy = np.array([])
    for e, row in enumerate(elements):
        if not e in ele_list: continue

        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

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
    sigma_xx_0 = sigma_xx[0] - (x[0]-0)*(sigma_xx[1]-sigma_xx[0])/(x[1]-x[0])
    sigma_yy_0 = sigma_yy[0] - (x[0]-0)*(sigma_yy[1]-sigma_yy[0])/(x[1]-x[0])

    fig, ax = plt.subplots(1,1)
    ax.scatter(x,sigma_xx, c='r', label="sigma_xx")
    ax.scatter(0,sigma_xx_0, edgecolors='r', facecolors='none')
    ax.scatter(x,sigma_yy, c='b', label="sigma_yy")
    ax.scatter(0,sigma_yy_0, edgecolors='b', facecolors='none')
    ax.set_xlabel("x/r")
    ax.set_ylabel(r"$\sigma/\sigma_{applied}$")
    ax.set_title("Stress along top wall")
    ax.legend()
    return fig, ax

def plot_stress_bottom(nodes_file, elements_file, stress, ele_list, index=0):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file)

    if index==1:
        elements[:,0:3]-=1
    
    i = np.arange(0,48,4)
    x = np.array([])
    sigma_xx = np.array([])
    sigma_yy = np.array([])
    for e, row in enumerate(elements):
        if not e in ele_list: continue

        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

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
    ax.scatter(1,sigma_xx_0, edgecolors='r', facecolors='none')
    ax.scatter(x,sigma_yy, c='b', label="sigma_yy")
    ax.scatter(1,sigma_yy_0, edgecolors='b', facecolors='none')
    ax.set_xlabel("x/r")
    ax.set_ylabel(r"$\sigma/\sigma_{applied}$")
    ax.set_title("Stress along bottom wall")
    ax.legend()
    return fig, ax

if __name__=="__main__":
    solveR()