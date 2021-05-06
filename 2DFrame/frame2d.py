import numpy as np
from scipy.linalg import solve

## Solve 2d frame problem
def solve_frame(nodes_file, elements_file, forces_file, disp_file, index=0):

    # Read and preallocate
    nodes, elements, forces, disp = read_inputs(nodes_file, elements_file, forces_file, disp_file, index)

    NN = nodes.shape[0]
    NE = elements.shape[0]
    Ndbcs = disp.shape[0]

    u = np.zeros((3*NN))
    F = np.zeros((3*NN))
    K = np.zeros((3*NN, 3*NN))
    eps = np.zeros((NE))
    stress = np.zeros((NE))
    Fi  = np.zeros((NE))
    Fe = np.zeros((3*NN))

    # Global Connectivity 
    # gcon(i, j) = global dof for node i and dof 
    gcon = np.zeros((NN, 3), dtype=int)
    for i in range(NN):
        gcon[i,0] = 3*i
        gcon[i,1] = 3*i+1
        gcon[i,2] = 3*i+2

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
        EA  = row[2] 
        EI  = row[3]

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]

        L = ((x2-x1)**2 + (y2-y1)**2)**0.5
        c = (x2-x1)/L
        s = (y2-y1)/L

        R = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        Kloc = np.array([
            [EA/L, 0, 0, -EA/L, 0, 0],
            [0, 12*EI/L**3, 6*EI/L**2, 0, -12*EI/L**3, 6*EI/L**2],
            [0, 6*EI/L**2,  4*EI/L,    0, -6*EI/L**2,  2*EI/L ],
            [-EA/L, 0, 0, EA/L, 0, 0],
            [0, -12*EI/L**3, -6*EI/L**2, 0, 12*EI/L**3, -6*EI/L**2],
            [0,  6*EI/L**2,   2*EI/L,    0, -6*EI/L**2,  4*EI/L ]
        ])

        Kloc = R.T@Kloc@R

        # Loop over local rows
        for inode in range(2):
            for idof in range(3):
                ldofi =  3*inode+idof
                gnodei = int(elements[e, inode])
                gdofi = gcon[gnodei, idof]

                # Loop over local columns
                for jnode in range(2):
                    for jdof in range(3):
                        ldofj = 3*jnode+jdof
                        gnodej = int(elements[e, jnode])
                        gdofj = gcon[gnodej, jdof]
                        K[gdofi,gdofj] += Kloc[ldofi,ldofj]

    # Reduce
    Kreduced = K
    Freduced = F

    for e, row in enumerate(elements):
        for inode in range(2):
            for idof in range(3):
                ldofi =  3*inode+idof
                gnodei = int(elements[e, inode])
                gdofi = gcon[gnodei, idof]

                # Check if dirichlet row
                if gdofi in dbcdof:
                    Kreduced[gdofi, :] = 0
                    Kreduced[gdofi, gdofi] = 1
                    Freduced[gdofi] = u[gdofi]
                    continue

                # Loop over local columns
                for jnode in range(2):
                    for jdof in range(3):
                        ldofj = 3*jnode+jdof
                        gnodej = int(elements[e, jnode])
                        gdofj = gcon[gnodej, jdof]

                        # If dirichlet column
                        if (gdofj in dbcdof):
                            Freduced[gdofi] -= K[gdofi,gdofj]*u[gdofj]

    # Solve for u
    u = solve(Kreduced, Freduced)

    # Postprocessing
    Fe = K@u
    # for e, row in enumerate(elements):
    #     n1 = int(row[0])
    #     n2 = int(row[1])
    #     E  = row[2] 
    #     I  = row[3]
    #     A  = row[3]

    #     x1, y1 = nodes[n1,0], nodes[n1,1]
    #     x2, y2 = nodes[n2,0], nodes[n2,1]
    #     u1, v1, a1 = u[gcon[n1,0]], u[gcon[n1,1]], u[gcon[n1,2]]
    #     u2, v2, a2 = u[gcon[n2,0]], u[gcon[n2,1]], u[gcon[n1,2]]

    #     L = ((x2-x1)**2 + (y2-y1)**2)**0.5
    #     c = (x2-x1)/L
    #     s = (y2-y1)/L

    #     eps[e] = (u2-u1)*c/L + (v2-v1)*s/L
    #     stress[e] = eps[e] * A
    #     Fi[e]  = eps[e]*E*A

    return u, Fe

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

