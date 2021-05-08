import numpy as np
from scipy.linalg import solve
from math import cos, sin, asin, pi

## Solve 3d frame problem
def solve_frame(nodes_file, elements_file, forces_file, disp_file, index=0):

    # Read and preallocate
    nodes, elements, forces, disp = read_inputs(nodes_file, elements_file, forces_file, disp_file, index)

    NN = nodes.shape[0]
    NE = elements.shape[0]
    Ndbcs = disp.shape[0]

    u = np.zeros((6*NN))
    F = np.zeros((6*NN))
    K = np.zeros((6*NN, 6*NN))
    eps = np.zeros((NE))
    stress = np.zeros((NE))
    Fi  = np.zeros((NE))
    Fe = np.zeros((6*NN))

    # Global Connectivity 
    # gcon(i, j) = global dof for node i and dof 
    gcon = np.arange(NN*6).reshape((NN,6))

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
        psi   = row[2] 
        EA    = row[3] 
        EI_Z  = row[4]
        EI_Y  = row[5]
        GI_P  = row[6]

        x1, y1, z1 = nodes[n1,0], nodes[n1,1], nodes[n1,2]
        x2, y2, z2 = nodes[n2,0], nodes[n2,1], nodes[n2,2]

        L = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
        cx = (x2-x1)/L
        cy = (y2-y1)/L
        cz = (z2-z1)/L
        theta = asin(cz)
        if theta==pi/2:
            phi=0
        else
            phi = asin(cy/cos(theta))

        c=cos(phi); s=sin(phi)
        R1 = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        c=cos(theta); s=sin(theta)
        R2 = np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])

        c=cos(psi); s=sin(psi)
        R3 = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

        R = R1@R2@R3
        R = R.T

        z = np.zeros((3,3))
        R = np.block([
            [R, z, z, z],
            [z, R, z, z],
            [z, z, R, z],
            [z, z, z, R]
        ])

        Kloc = np.zeros((12,12))
        Kloc[0,0]=EA/L; Kloc[0,6]=EA/L
        Kloc[1,1]=12*EI_Z/L**3; Kloc[1,5]=6*EI_Z/L**2; Kloc[1,7]=-12*EI_Z/L**3; Kloc[1,11]=6*EI_Z/L**2
        Kloc[2,2]=12*EI_Y/L**3; Kloc[2,4]=-6*EI_Y/L**2; Kloc[2,8]=-12*EI_Y/L**3; Kloc[2,10]=-6*EI_Y/L**2
        Kloc[3,3]=GI_P/L; Kloc[3,9]=GI_P/L
        Kloc[4,4]=4*EI_Y/L; Kloc[4,8]=6*EI_Y/L**2; Kloc[4,10]=2*EI_Y/L
        Kloc[5,5]=4*EI_Z/L; Kloc[5,7]=-6*EI_Z/L**2; Kloc[5,11]=2*EI_Z/L
        Kloc[6,6]=EA/L
        Kloc[7,7]=12*EI_Z/L**3; Kloc[7,11]=-6*EI_Y/L**2
        Kloc[8,8]=12*EI_Y/L**3; Kloc[8,10]=6*EI_Y/L**2
        Kloc[9,9]=GI_P/L
        Kloc[10,10]=4*EI_Y/L
        Kloc[11,11]=4*EI_Z/L

        Kloc = Kloc + Kloc.T - np.diag(Kloc.diagonal())
        Kloc = R.T@Kloc@R

        # Loop over local rows
        for inode in range(2):
            for idof in range(6):
                ldofi =  6*inode+idof
                gnodei = int(elements[e, inode])
                gdofi = gcon[gnodei, idof]

                # Loop over local columns
                for jnode in range(2):
                    for jdof in range(6):
                        ldofj = 6*jnode+jdof
                        gnodej = int(elements[e, jnode])
                        gdofj = gcon[gnodej, jdof]
                        K[gdofi,gdofj] += Kloc[ldofi,ldofj]

    # Reduce
    Kreduced = K
    Freduced = F

    for e, row in enumerate(elements):
        for inode in range(2):
            for idof in range(6):
                ldofi =  6*inode+idof
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
                    for jdof in range(6):
                        ldofj = 6*jnode+jdof
                        gnodej = int(elements[e, jnode])
                        gdofj = gcon[gnodej, jdof]

                        # If dirichlet column
                        if (gdofj in dbcdof):
                            Freduced[gdofi] -= K[gdofi,gdofj]*u[gdofj]

    # Solve for u
    u = solve(Kreduced, Freduced)

    # Find nodal forces
    Fe = K@u

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

