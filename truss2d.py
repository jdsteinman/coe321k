import numpy as np
from scipy.linalg import solve

## Solve 2d truss problem
def truss2d(nodes_file, elements_file, forces_file, disp_file, index=0):

    # Read and preallocate
    nodes, elements, forces, disp = read_inputs(nodes_file, elements_file, forces_file, disp_file, index)

    NN = nodes.shape[0]
    NE = elements.shape[0]
    Ndbcs = disp.shape[0]

    u = np.zeros((2*NN))
    F = np.zeros((2*NN))
    Kloc = np.zeros((4,4))
    K = np.zeros((2*NN, 2*NN))
    eps = np.zeros((NE))
    stress = np.zeros((NE))
    Fi  = np.zeros((NE))
    Fe = np.zeros((2*NN))

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
    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        E  = row[2] 
        A  = row[3]

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]

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
                gnodei = int(elements[e, inode])
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
                        gnodej = int(elements[e, jnode])
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
        n1 = int(row[0])
        n2 = int(row[1])
        E  = row[2] 
        A  = row[3]

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        u1, v1 = u[gcon[n1,0]], u[gcon[n1,1]]
        u2, v2 = u[gcon[n2,0]], u[gcon[n2,1]]

        L = ((x2-x1)**2 + (y2-y1)**2)**0.5
        c = (x2-x1)/L
        s = (y2-y1)/L

        eps[e] = (u2-u1)*c/L + (v2-v1)*s/L
        stress[e] = eps[e] * A
        Fi[e]  = eps[e]*E*A
        Fe[gcon[n1,0]] -= Fi[e] * c
        Fe[gcon[n1,1]] -= Fi[e] * s
        Fe[gcon[n2,0]] += Fi[e] * c
        Fe[gcon[n2,1]] += Fi[e] * s

    return u, eps, stress, Fi, Fe

## Inputs
def read_inputs(nodes, elements, forces, disp, index=0):
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

    if index==0:
        pass
    elif index==1:
        elements[:,0:2]-=1
        disp[:,0:2]-=1
        forces[:,0:2]-=1
    else:
        print("Invalid file ordering: " + index)
        return

    return nodes, elements, forces, disp


