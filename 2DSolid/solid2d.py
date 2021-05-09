import numpy as np
from scipy.linalg import solve

## Solve 2d frame problem
def solve_solid(nodes_file, elements_file, forces_file, disp_file, index=0):

    # Read and preallocate
    nodes = read_nodes(nodes_file)
    elements, E, nu = read_elements(elements_file, index)
    disp = read_disp(disp_file, index)
    forces = read_forces(forces_file, index)

    NN = nodes.shape[0]
    NE = elements.shape[0]
    Ndbcs = disp.shape[0]

    u = np.zeros((2*NN))
    F = np.zeros((2*NN))
    K = np.zeros((2*NN, 2*NN))
    B = np.zeros((NE, 3, 6))
    strain = np.zeros((NE, 3))
    stress = np.zeros((NE, 3))
    Fi  = np.zeros((NE))
    Fe = np.zeros((2*NN))

    C = np.array([
            [E/(1-nu*nu), nu*E/(1-nu*nu), 0],
            [nu*E/(1-nu*nu), E/(1-nu*nu), 0],
            [0, 0, 0.5*E/1+nu]
    ])  

    # Global Connectivity 
    # gcon(i, j) = global dof for node i and dof 
    gcon = np.arange(2*NN).reshape(-1,2)

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
        n3 = int(row[2])

        x1, y1 = nodes[n1,0], nodes[n1,1]
        x2, y2 = nodes[n2,0], nodes[n2,1]
        x3, y3 = nodes[n3,0], nodes[n3,1]

        # Element area
        A = 0.5*(-x2*y1+x3*y1+x1*y2-x3*y2-x1*y3+x2*y3)

        # B matrix
        Bloc = np.array([
            [y2-y3, 0, -y1+y3, 0, y1-y2, 0],
            [0, -x2+x3, 0, x1-x3, 0, -x1+x2],
            [-x2+x3, y2-y3, x1-x3, -y1+y3, -x1+x2, y1-y2]
        ])
        Bloc = Bloc/(2*A)
        B[e] = Bloc     

        Kloc = A*Bloc.T@C@Bloc

        # Loop over local rows
        for inode in range(3):
            for idof in range(2):
                ldofi =  2*inode+idof
                gnodei = int(elements[e, inode])
                gdofi = gcon[gnodei, idof]

                # Loop over local columns
                for jnode in range(3):
                    for jdof in range(2):
                        ldofj = 2*jnode+jdof
                        gnodej = int(elements[e, jnode])
                        gdofj = gcon[gnodej, jdof]
                        K[gdofi,gdofj] += Kloc[ldofi,ldofj]

    # Reduce
    Kreduced = K
    Freduced = F

    for e, row in enumerate(elements):
        for inode in range(3):
            for idof in range(2):
                ldofi =  2*inode+idof
                gnodei = int(elements[e, inode])
                gdofi = gcon[gnodei, idof]

                # Check if dirichlet row
                if gdofi in dbcdof:
                    Kreduced[gdofi, :] = 0
                    Kreduced[gdofi, gdofi] = 1
                    Freduced[gdofi] = u[gdofi]
                    continue

                # Loop over local columns
                for jnode in range(3):
                    for jdof in range(2):
                        ldofj = 2*jnode+jdof
                        gnodej = int(elements[e, jnode])
                        gdofj = gcon[gnodej, jdof]

                        # If dirichlet column
                        if (gdofj in dbcdof):
                            Freduced[gdofi] -= K[gdofi,gdofj]*u[gdofj]

    # Solve for u
    u = solve(Kreduced, Freduced)

    # Find nodal forces
    Fe = K@u

    for e, row in enumerate(elements):
        n1 = int(row[0])
        n2 = int(row[1])
        n3 = int(row[2])

        index = [[n1,n1,n2,n2,n3,n3],[0,1,0,1,0,1]]
        u_ele = u[gcon[index]]
        u_ele = u_ele.reshape((-1,1))

        eps  = B[e]@u_ele
        sigma = C@eps

        strain[e] = eps.ravel()
        stress[e] = sigma.ravel()

    return u, strain, stress


def read_nodes(nodes_file):
    nodes = []
    with open(nodes_file, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                nums = line.split()
                NN = int(nums[0])
            else:
                nums = line.split()
                x = float(nums[1])
                y = float(nums[2])
                nodes.append([x, y])
    nodes = np.array(nodes)          
    return nodes

def read_elements(elements_file, index):
    elements = []
    with open(elements_file, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                nums = line.split()
                NE = int(nums[0])
                E = float(nums[1])
                nu = float(nums[2])
            else:
                nums = line.split()
                n1 = float(nums[1])
                n2 = float(nums[2])
                n3 = float(nums[3])
                elements.append([n1, n2, n3])

    elements = np.array(elements)     
    if index==1:
        elements[:,0:3]-=1

    return elements, E, nu

def read_disp(disp_file, index):
    disp = []
    with open(disp_file, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                nums = line.split()
                ND = int(nums[0])
            else:
                nums = line.split()
                node = int(nums[0])
                dof = int(nums[1])
                val = float(nums[2])
                disp.append([node, dof, val])
    disp = np.array(disp)     
    if index==1:
        disp[:,0:2]-=1

    return disp

def read_forces(forces_file, index):
    forces = []
    with open(forces_file, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                nums = line.split()
                NF = int(nums[0])
            else:
                nums = line.split()
                node = int(nums[0])
                dof = int(nums[1])
                val = float(nums[2])
                forces.append([node, dof, val])
    forces = np.array(forces)  
    if index==1:
        forces[:,0:2]-=1   

    return forces