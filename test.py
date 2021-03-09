import numpy as np
from scipy.linalg import solve

## Solve 2d truss problem
def truss2d(nodes_file, elements_file, forces_file, disp_file):

    # Read and preallocate
    nodes, elements, forces, disp = read_inputs(nodes_file, elements_file, forces_file, disp_file)

    NN = nodes.shape[0]
    NE = elements.shape[0]
    Ndbcs = disp.shape[0]
    Ndofs = 2*NN-Ndbcs

    u = np.zeros((2*NN))
    F = np.zeros((2*NN))
    Kloc = np.zeros((4,4))
    K = np.zeros((2*NN, 2*NN))
    eps = np.zeros((NE, 1))
    Fi  = np.zeros((NN, 1))
    Fcheck = np.zeros((2*NN))

    # Global Connectivity 
    # gcon(i, j) = global dof for node i and dof 
    gcon = np.zeros((NN, 2), dtype=int)
    for i in range(NN):
        gcon[i,0] = 2*i
        gcon[i,1] = 2*i+1

    # Assign Dirichlet Displacement
    dbcnode = disp[:,0].astype(int)
    dbcdof = disp[:,1].astype(int)
    dbcval = disp[:,2]
    dbcdof = gcon[dbcnode, dbcdof]
    u[dbcdof] = dbcval

    # Assign Force conditions
    fnode = forces[:,0].astype(int)
    fdof = forces[:,1].astype(int)
    fval = forces[:,2]
    fdof = gcon[fnode, fdof]
    F[fdof] = fval

    # Stiffness assembly
    econ = elements[:,1:3]
    for e, row in enumerate(elements):
        n1 = int(row[1])
        n2 = int(row[2])
        E  = row[3] 
        A  = row[4]

        x1, y1 = nodes[n1,1], nodes[n1,2]
        x2, y2 = nodes[n2,1], nodes[n2,2]

        L = ((x2-x1)**2 + (y2-y1)**2)**0.5
        c = (x2-x1)/L
        s = (y2-y1)/L

        # Local stiffness for element
        k1 = E*A/L*np.array([ c*c,  c*s, -c*c, -c*s])
        k2 = E*A/L*np.array([ c*s,  s*s, -c*s, -s*s])
        k3 = E*A/L*np.array([-c*c, -c*s,  c*c,  c*s])
        k4 = E*A/L*np.array([-c*s, -s*s,  c*s,  s*s])

        Kloc = np.array([k1, k2, k3, k4])

        # Loop over local rows
        for inode in range(2):
            for idof in range(2):
                ldofi =  2*inode+idof
                gnodei = int(econ[e, inode])
                gdofi = gcon[gnodei, idof]

                # Check if dirichlet row
                if gdofi in dbcdof:
                    K[gdofi, gdofi] = 1
                    F[gdofi] = u[gdofi]
                    continue

                # Loop over local columns
                for jnode in range(2):
                    for jdof in range(2):
                        ldofj = 2*jnode+jdof
                        gnodej = int(econ[e, jnode])
                        gdofj = gcon[gnodej, jdof]

                        # If dirichlet column
                        if (gdofj in dbcdof):
                            F[gdofi] -= Kloc[ldofi,ldofj]*u[gdofj]
                        else:
                            K[gdofi,gdofj] += Kloc[ldofi,ldofj]


    # Solve for u
    u = solve(K, F)

    # Solve for Strains
    for e, row in enumerate(elements):
        n1 = int(row[1])
        n2 = int(row[2])
        E  = row[3] 
        A  = row[4]

        x1, y1 = nodes[n1,1], nodes[n1,2]
        x2, y2 = nodes[n2,1], nodes[n2,2]
        u1, v1 = u[gcon[n1,0]], u[gcon[n1,1]]
        u2, v2 = u[gcon[n2,0]], u[gcon[n2,1]]

        L = ((x2-x1)**2 + (y2-y1)**2)**0.5
        c = (x2-x1)/L
        s = (y2-y1)/L

        eps[e] = (u2-u1)*c/L + (v2-v1)*s/L
        Fi[e]  = eps[e]*E*A
        Fcheck[gcon[n1,0]] -= Fi[e] * c
        Fcheck[gcon[n1,1]] -= Fi[e] * s
        Fcheck[gcon[n2,0]] += Fi[e] * c
        Fcheck[gcon[n2,1]] += Fi[e] * s

    np.set_printoptions(precision=10)
    print("u: \n", u)
    print("eps: \n", eps)
    print("Fi: \n", Fi)
    print("Fcheck: \n", Fcheck)

    return u

## Inputs
def read_inputs(nodes, elements=None, forces=None, disp=None):
    nodes = np.genfromtxt(nodes, comments='#')
    elements = np.genfromtxt(elements, comments='#')
    forces = np.genfromtxt(forces, comments='#')
    disp = np.genfromtxt(disp, comments='#')

    if elements.ndim == 1:
        elements = elements.reshape((1, -1))
    if disp.ndim == 1:
        disp = disp.resape((1, -1))
    if forces.ndim == 1:
        forces = forces.reshape((1,-1))

    nodes[:,0]-=1
    elements[:,0:3]-=1
    disp[:,0:2]-=1
    forces[:,0:2]-=1

    return nodes, elements, forces, disp

## Test function
path = "./test/"
nodes = path + "nodes.txt"
elements = path + "elements.txt"
forces = path + "forces.txt"
disp = path + "disp.txt"

truss2d(nodes, elements, forces, disp)

