import numpy as np
import matplotlib.pyplot as plt
from solid2d import *


def main():
    tag = ['6','12','24','R']

    fig, ax = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig3, ax3 = plt.subplots(1,1)
    fig4, ax4 = plt.subplots(1,1)

    colors = ['b','r','g','orange']
    labels = ['Ex6','Ex12','Ex24','R']

    for i in range(4):
        nodes = "inputs/nodes"+tag[i]+".txt"
        elements = "inputs/elements"+tag[i]+".txt"
        stress  = np.loadtxt("output/stress"+tag[i]+".txt")

        plot_stress_left(ax, stress, 0, colors[i], labels[i], nodes, elements)
        plot_stress_left(ax2, stress, 1, colors[i], labels[i], nodes, elements)
        plot_stress_bottom(ax3, stress, 0, colors[i], labels[i], nodes, elements)
        plot_stress_bottom(ax4, stress, 1, colors[i], labels[i], nodes, elements)

    plt.show()


def plot_stress_left(ax, stress, ii, color, label, nodes_file, elements_file):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file, 1)
    
    y = np.array([])
    sigma = np.array([])
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
        sigma = np.append(sigma, stress[e,ii])

    # sort
    ind = np.argsort(y)
    y = y[ind]
    sigma = sigma[ind]

    # Extrapolate 
    sigma_0 = sigma[0] - (y[0]-1)*(sigma[1]-sigma[0])/(y[1]-y[0])

    ax.scatter(y, sigma, c=color, label=label)
    ax.scatter(1, sigma_0, edgecolors=color, facecolors='none')


def plot_stress_bottom(ax, stress, ii, color, label, nodes_file, elements_file):
    nodes = read_nodes(nodes_file)
    elements, _, _ = read_elements(elements_file, 1)
    
    x = np.array([])
    sigma = np.array([])
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
        sigma = np.append(sigma, stress[e,ii])

    # sort
    ind = np.argsort(x)
    x = x[ind]
    sigma = sigma[ind]

    # Extrapolate 
    sigma_0 = sigma[0] - (x[0]-1)*(sigma[1]-sigma[0])/(x[1]-x[0])

    ax.plot()
    ax.scatter(x,sigma, c=color, label=label)
    ax.scatter(1,sigma_0, edgecolors=color, facecolors='none')

if __name__=="__main__":
    main()