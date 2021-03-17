import numpy as np
import matplotlib.pyplot as plt
from truss2d import truss2d, read_inputs
from math import pi

path = "./inputs4/"
nodes = path + "nodes.txt"
elements = path + "elements.txt"
forces = path + "fbcs.txt"
disp = path + "dbcs.txt"

u, eps, stress, Fi, Fe = truss2d(nodes, elements, forces, disp, index=1)
print('{:^8} {:^20}'.format("element","internal force (GN)"))
for i, val in enumerate(Fi):
    print('{:^8d} {:< 15f}'.format(i, val))

## Part 2: Stress analysis
nodes, elements, fbc, dbc = read_inputs(nodes, elements, forces, disp, index=1)
NE = elements.shape[0]

yieldStrength = 250 / 1000 # since input files are scales to GPa and 1 GPa = 1000 MPa

maxStress = np.max(stress)
print("Critical Tension Element:")
print('Maximum Beam stress:', maxStress,'(GPa) in beam', np.argmax(stress)+1)

yieldCritLoad = yieldStrength/maxStress * abs(fbc[0,2])
print('Critical Yielding load: ', yieldCritLoad,'(GN)\n')

# Wide flange bar dimensions (mm):
a = 25  / 1000
b = 200 / 1000
h = 200 / 1000
H = 225 / 1000

Ix = a*(h**3)/12 + b*(H**3 - h**3)/12
Iy = (a**3)*h/12 + (b**3)*(H - h)/12
I = min(Ix,Iy)

# Buckling
Ncr = np.zeros((NE))
P =  np.zeros((NE))
for i, row in enumerate(elements):
    if Fi[i] > 0:
        continue
       
    E = row[2]
   
    # local nodes
    x1 = nodes[int(row[0]),0]
    y1 = nodes[int(row[0]),1]
    x2 = nodes[int(row[1]),0]
    y2 = nodes[int(row[1]),1]

    # calculate length of element and angular values
    L = ((x2-x1)**2 + (y2-y1)**2)**0.5  
   
    Ncr = (pi**2)*E*I / L**2
    if Fi[i] < 0:
        P[i] = Ncr/Fi[i]
    else:
        P[i] = 0
   
print("Critical Compression Element: ")
print("Critical Buckling Load: ", np.amin(P), "(GN) on element ", np.argmin(P))

fig, ax = plt.subplots(1,1)
ax.set_aspect('equal', adjustable='box')
for row in elements:
    x1 = nodes[int(row[0]),0]
    y1 = nodes[int(row[0]),1]
    x2 = nodes[int(row[1]),0]
    y2 = nodes[int(row[1]),1]

    ax.plot([x1,x2],[y1,y2])


fig2, ax2 = plt.subplots(1,1)
ax2.set_aspect('equal', adjustable='box')
u = u.reshape((-1,2))
nodes = nodes + u
for row in elements:
    x1 = nodes[int(row[0]),0]
    y1 = nodes[int(row[0]),1]
    x2 = nodes[int(row[1]),0]
    y2 = nodes[int(row[1]),1]

    ax2.plot([x1,x2],[y1,y2])

# plt.show()

