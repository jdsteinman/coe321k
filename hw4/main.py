import numpy as np
import matplotlib.pyplot as plt
from truss2d import truss2d, read_inputs
from math import pi

path = "./inputs/"
nodes = path + "nodes.txt"
elements = path + "elements.txt"
forces = path + "fbcs.txt"
disp = path + "dbcs.txt"

u, eps, stress, Fi, Fe = truss2d(nodes, elements, forces, disp, index=1)
print("u: \n", u)
print("eps: \n", eps)
print("Fi: \n", Fi)
print("Fe: \n", Fe)

## Part 2: Stress analysis
nodes, elements, fbc, dbc = read_inputs(nodes, elements, forces, disp, index=1)
print(fbc)
NE = elements.shape[0]

yieldStrength = 250 / 1000 # since input files are scales to GPa and 1 GPa = 1000 MPa

print(stress)
maxBarStress = np.max(stress)
print('The maximum beam stress is', maxBarStress,'(P/m^2) in beam', np.argmax(stress)+1)

yieldCritLoad = yieldStrength/maxBarStress
print('Therefore, with beams made of Structural ASTM-A36 Steel, the critical load to cause yielding is', yieldCritLoad,'(P)')

# Wide flange bar dimensions (mm):
a = 25  / 1000
b = 200 / 1000
h = 200 / 1000
H = 225 / 1000

Ix = a*(h**3)/12 + b*(H**3 - h**3)/12
Iy = (a**3)*h/12 + (b**3)*(H - h)/12
I = min(Ix,Iy)

Ncr = np.zeros((NE))
for i, element in enumerate(elements):
    if Fi[i] > 0:
        continue
       
    E = element[2]
   
    # detemine nodes of each element
    x1 = nodes[int(element[0]),0]
    y1 = nodes[int(element[0]),1]
    x2 = nodes[int(element[1]),0]
    y2 = nodes[int(element[1]),1]

    # calculate length of element and angular values
    L = ((x2-x1)**2 + (y2-y1)**2)**0.5  
   
    Ncr[i] = (pi**2)*E*I / L**2
    print(Ncr[i])
   
    Ptemp = Ncr[i]/abs(Fi[i])
   
    if i == 0:
        Pmin = Ptemp
   
    elif Ptemp < Pmin:
        Pmin = Ptemp
   
print(Pmin)

u = u.reshape((-1,2))
nodes_d = nodes + u

fig, ax = plt.subplots(1,1)
ax.set_aspect('equal', adjustable='box')
# ax.scatter(nodes[:,0], nodes[:,1])

for row in elements:
    x1 = nodes_d[int(row[0]),0]
    y1 = nodes_d[int(row[0]),1]
    x2 = nodes_d[int(row[1]),0]
    y2 = nodes_d[int(row[1]),1]

    # x1 = nodes[int(row[0]),0]
    # y1 = nodes[int(row[0]),1]
    # x2 = nodes[int(row[1]),0]
    # y2 = nodes[int(row[1]),1]

    ax.plot([x1,x2],[y1,y2])
    ax.set_aspect('equal', adjustable='box')

ax.set
plt.show()