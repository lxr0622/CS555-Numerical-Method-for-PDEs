
#Read in the mesh:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla
import matplotlib.tri as tri
import gmsh
mesh = gmsh.Mesh()
mesh.read_msh('wdg3.msh')

# Mesh |  ne   |  nv
#  w1  |  255  |  557
#  w2  |  1020 |  2134
#  w3  |  4080 |  8348

## Quadratic triangle elements
E = mesh.Elmts[9][1]
V = mesh.Verts[:,:2]

ne = E.shape[0]
nv = V.shape[0]
X = V[:,0]
Y = V[:,1]
print(X)

def checkorientation(V, E):
    sgn = np.zeros((E.shape[0],))
    for i in range(E.shape[0]):
        xi = V[E[i, :3],0]
        yi = V[E[i, :3],1]
        A = np.zeros((3,3))
        A[:,0] = 1.0
        A[:,1] = xi
        A[:,2] = yi
        sgn[i] = np.linalg.det(A)
    return sgn

sgn = checkorientation(V, E)
I = np.where(sgn<0)[0]
if(I.size==ne or I.size==0):
    print('all elements have consistent orientation')

# plt.figure(figsize=(19,10))
plt.figure(figsize=(8,5))
plt.triplot(X,Y,E[:,:3])
plt.plot(X,Y,'kx')
# plt.tricontourf(X, Y, E[:,:3], (X-5)**2 + (Y-5)**2, 100)
# plt.colorbar()
plt.show()


#Exact SOlution and Plot:
def wan_exact(): # Wannier-Stokes
    d = 2.
    r = 1.
    s = np.sqrt(d**2 - r**2)
    gam = (s+d)/(d-s)
    U = 4.
    tmp = 1./np.log(gam)
    A = -(U*d)* tmp
    B = 2.*U*(d+s) * tmp
    C = 2.*U*(d-s) * tmp
    F = U * tmp
    
    from sympy import symbols, exp, log, lambdify, init_printing, diff, simplify
    x,y,k1,k2,u,v=symbols('x y k1 k2 u v')
    k1 = x**2 + (s+y)**2
    k2 = x**2 + (s-y)**2
    
    # # maslanik, sani, gresho
    u = U - F*log(k1/k2)\
          - 2*(A+F*y) *((s+y)+k1*(s-y)/k2) / k1\
          - B*((s+2*y)-2*y*((s+y)**2)/k1)/k1\
          - C*((s-2*y)+2*y*((s-y)**2)/k2)/k2

    v = + 2*x*(A+F*y)*(k2-k1)/(k1*k2)\
        - 2*B*x*y*(s+y)/(k1*k1)\
        - 2*C*x*y*(s-y)/(k2*k2)
        
    p = - 4*B*x*(s+y)/(k1*k1)\
        - 4*C*x*(s-y)/(k2*k2)\
        - 16*F*s*x*y/(k1*k2)

    ue = lambdify([x,y],u,'numpy')
    ve = lambdify([x,y],v,'numpy')
    pe = lambdify([x,y],p,'numpy')
    return ue,ve,pe


ue, ve, pe = wan_exact()
uex = ue(X,Y)
vex = ve(X,Y)
#pex = pe(X1,Y1)

#Build your matrix:  
#Gauss Point:
def trigauss(n):
    if (n == 1):
        xw=np.array([0.33333333333333, 0.33333333333333, 1.00000000000000])
    elif (n == 2):
        xw=np.array([[0.16666666666667, 0.16666666666667, 0.33333333333333],
                     [0.16666666666667, 0.66666666666667, 0.33333333333333],
                     [0.66666666666667, 0.16666666666667, 0.33333333333333]])
    elif (n == 3):
        xw=np.array([[0.33333333333333, 0.33333333333333, -0.56250000000000],
                     [0.20000000000000, 0.20000000000000, 0.52083333333333],
                     [0.20000000000000, 0.60000000000000, 0.52083333333333],
                     [0.60000000000000, 0.20000000000000, 0.52083333333333]])
    elif (n == 4):
        xw=np.array([[0.44594849091597, 0.44594849091597, 0.22338158967801],
                     [0.44594849091597, 0.10810301816807, 0.22338158967801],
                     [0.10810301816807, 0.44594849091597, 0.22338158967801],
                     [0.09157621350977, 0.09157621350977, 0.10995174365532],
                     [0.09157621350977, 0.81684757298046, 0.10995174365532],
                     [0.81684757298046, 0.09157621350977, 0.10995174365532]])
    elif (n == 5):
        xw=np.array([[0.33333333333333, 0.33333333333333, 0.22500000000000],
                     [0.47014206410511, 0.47014206410511, 0.13239415278851],
                     [0.47014206410511, 0.05971587178977, 0.13239415278851],
                     [0.05971587178977, 0.47014206410511, 0.13239415278851],
                     [0.10128650732346, 0.10128650732346, 0.12593918054483],
                     [0.10128650732346, 0.79742698535309, 0.12593918054483],
                     [0.79742698535309, 0.10128650732346, 0.12593918054483]])
    elif (n == 6):
        xw=np.array([[0.24928674517091, 0.24928674517091, 0.11678627572638 ],
                     [0.24928674517091, 0.50142650965818, 0.11678627572638 ],
                     [0.50142650965818, 0.24928674517091, 0.11678627572638 ],
                     [0.06308901449150, 0.06308901449150, 0.05084490637021 ],
                     [0.06308901449150, 0.87382197101700, 0.05084490637021 ],
                     [0.87382197101700, 0.06308901449150, 0.05084490637021 ],
                     [0.31035245103378, 0.63650249912140, 0.08285107561837 ],
                     [0.63650249912140, 0.05314504984482, 0.08285107561837 ],
                     [0.05314504984482, 0.31035245103378, 0.08285107561837 ],
                     [0.63650249912140, 0.31035245103378, 0.08285107561837 ],
                     [0.31035245103378, 0.05314504984482, 0.08285107561837 ],
                     [0.05314504984482, 0.63650249912140, 0.08285107561837]])
    elif (n == 7):
        xw=np.array([[0.33333333333333, 0.33333333333333, -0.14957004446768],
                     [0.26034596607904, 0.26034596607904, 0.17561525743321 ],
                     [0.26034596607904, 0.47930806784192, 0.17561525743321 ],
                     [0.47930806784192, 0.26034596607904, 0.17561525743321 ],
                     [0.06513010290222, 0.06513010290222, 0.05334723560884 ],
                     [0.06513010290222, 0.86973979419557, 0.05334723560884 ],
                     [0.86973979419557, 0.06513010290222, 0.05334723560884 ],
                     [0.31286549600487, 0.63844418856981, 0.07711376089026 ],
                     [0.63844418856981, 0.04869031542532, 0.07711376089026 ],
                     [0.04869031542532, 0.31286549600487, 0.07711376089026 ],
                     [0.63844418856981, 0.31286549600487, 0.07711376089026 ],
                     [0.31286549600487, 0.04869031542532, 0.07711376089026 ],
                     [0.04869031542532, 0.63844418856981, 0.07711376089026]])
    elif (n >= 8):
        if(n>8):
            print('trigauss: Too high, taking n = 8')
        xw=np.array([[0.33333333333333, 0.33333333333333, 0.14431560767779],
                     [0.45929258829272, 0.45929258829272, 0.09509163426728],
                     [0.45929258829272, 0.08141482341455, 0.09509163426728],
                     [0.08141482341455, 0.45929258829272, 0.09509163426728],
                     [0.17056930775176, 0.17056930775176, 0.10321737053472],
                     [0.17056930775176, 0.65886138449648, 0.10321737053472],
                     [0.65886138449648, 0.17056930775176, 0.10321737053472],
                     [0.05054722831703, 0.05054722831703, 0.03245849762320],
                     [0.05054722831703, 0.89890554336594, 0.03245849762320],
                     [0.89890554336594, 0.05054722831703, 0.03245849762320],
                     [0.26311282963464, 0.72849239295540, 0.02723031417443],
                     [0.72849239295540, 0.00839477740996, 0.02723031417443],
                     [0.00839477740996, 0.26311282963464, 0.02723031417443],
                     [0.72849239295540, 0.26311282963464, 0.02723031417443],
                     [0.26311282963464, 0.00839477740996, 0.02723031417443],
                     [0.00839477740996, 0.72849239295540, 0.02723031417443]])

    qx = xw[:,:2]
    qw = xw[:,2]/2
    return qx, qw

#Node Mapping:
E1,Eindex = np.unique(E[:,:3],return_index=True)
index_map = np.zeros(int(np.amax(E1)+1))
index_map[E1] = np.arange(len(E1))
index_map = index_map.astype(int)
X1 = V[E1,0]
Y1 = V[E1,1]

E2 = np.zeros(E[:,:3].shape)


#Choose for Gauss Point:
qx,qw=trigauss(5)
#Matrix:
#Define the function:
def dbasis_fun(r,s):
    dbasis=np.array([[4*s+4*r-3, 4*r-1, 0,4-8*r-4*s,4*s,-4*s],
              [4*s+4*r-3, 0, 4*s-1,-4*r,4*r,4-4*r-8*s]])
    return dbasis
def basis_fun(r,s):
    basis=np.array([(1-r-s)*(1-2*r-2*s),-r*(1-2*r),-s*(1-2*s),4*r*(1-r-s),4*r*s,4*s*(1-r-s)])
    return basis   

#def dbasis_funp(r,s):
#    dbasisp=np.array([[4*s+4*r-3, 4*r-1, 0],
#              [4*s+4*r-3, 0, 4*s-1]])
#    return dbasisp
def basis_funp(r,s):
    basisp=np.array([1.0-r-s,r,s])
    return basisp   
#created space for the data, row, and column indices of the matrix (AA, IA, JA) and of the right-hand side (bb, ib,jb)
AA = np.zeros((ne, 36))
IA = np.zeros((ne, 36))
JA = np.zeros((ne, 36))

Dx = np.zeros((ne, 18))
IDx = np.zeros((ne, 18))
JDx = np.zeros((ne, 18))

Dy = np.zeros((ne, 18))
IDy = np.zeros((ne, 18))
JDy = np.zeros((ne, 18))


for ei in range(E.shape[0]):
    E2[ei,:] = index_map[E[ei,:3]]

for ei in range(0, ne):
    # Step 1, coordinate:
    K = E[ei, :]
    x0, y0 = X[K[0]], Y[K[0]]
    x1, y1 = X[K[1]], Y[K[1]]
    x2, y2 = X[K[2]], Y[K[2]]


    # Step 2: Jacobian Matrix
    J = np.array([[x1 - x0, x2 - x0],
                  [y1 - y0, y2 - y0]])
    invJ = la.inv(J.T)
    detJ = la.det(J)

    # Step 3: basis function or derivative of function:
    dbasis=[]
    basis=[]
    basisp=[]
    for i in range(np.shape(qx)[0]):
        dbasis.append(dbasis_fun(qx[i,0],qx[i,1]))
        basis.append( basis_fun(qx[i,0],qx[i,1]))
        #dbasisp.append(dbasis_funp(qx[i,0],qx[i,1]))
        basisp.append( basis_funp(qx[i,0],qx[i,1]))

    # Step 4
    dphi=[]
    #dphip=[]
    for i in range(np.shape(qx)[0]):
        dphi.append(invJ.dot(dbasis[i]))
        #dphip.append(invJ.dot(dbasisp[i]))

    # Step 5 , A element,Dx element, Dy element:
    Aelem=np.zeros((6,6))
    for i in range(np.shape(qx)[0]):
        Aelem=Aelem+detJ*(qw[i]*(dphi[i].T).dot(dphi[i]))
#        print('shape',basisp[i],np.shape(basisp[i]))
#        print('shape',dphi[i][0,:],np.shape(dphi[i][0,:]))
    Dxelem=np.zeros((3,6))
    Dyelem=np.zeros((3,6))
    #dbasisp=[]
    for i in range(np.shape(qx)[0]):
        Dxelem=Dxelem+detJ*qw[i]*np.outer( basisp[i],(dphi[i].T[:,0]))
        Dyelem=Dyelem+detJ*qw[i]*np.outer( basisp[i],(dphi[i].T[:,1]))
    
 
    #Step 8, Assembly A and Matrix D
    AA[ei, :] = Aelem.ravel()
    IA[ei, :] = [K[0], K[0], K[0],K[0], K[0], K[0],K[1], K[1], K[1], K[1], K[1], K[1], 
                 K[2], K[2], K[2],K[2], K[2], K[2],K[3], K[3], K[3],K[3], K[3], K[3],
                 K[4],K[4],K[4],K[4],K[4],K[4],K[5],K[5],K[5],K[5],K[5],K[5]]
    
    JA[ei, :] = [K[0], K[1], K[2],K[3], K[4], K[5], 
                 K[0], K[1], K[2],K[3], K[4], K[5], 
                 K[0], K[1], K[2],K[3], K[4], K[5],
                 K[0], K[1], K[2],K[3], K[4], K[5],
                 K[0], K[1], K[2],K[3], K[4], K[5],
                 K[0], K[1], K[2],K[3], K[4], K[5]]
    
    Dx[ei,:]=Dxelem.ravel()
    Dy[ei,:]=Dyelem.ravel()
    IDx[ei,:]=index_map[[K[0], K[0], K[0],K[0], K[0], K[0],K[1], K[1], K[1], K[1], K[1], K[1],
                 K[2], K[2], K[2], K[2], K[2], K[2]]]
    IDy[ei,:]=index_map[[K[0], K[0], K[0],K[0], K[0], K[0],K[1], K[1], K[1], K[1], K[1], K[1],
                 K[2], K[2], K[2], K[2], K[2], K[2]]]
    
    JDx[ei, :] = [K[0], K[1], K[2],K[3], K[4], K[5], 
                 K[0], K[1], K[2],K[3], K[4], K[5], 
                 K[0], K[1], K[2],K[3], K[4], K[5]]
    
    JDy[ei, :] =[K[0], K[1], K[2],K[3], K[4], K[5], 
                 K[0], K[1], K[2],K[3], K[4], K[5], 
                 K[0], K[1], K[2],K[3], K[4], K[5]]

#convert back to COO for easier manipulation.
Abar = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
Abar = Abar.tocsr()
Abar = Abar.tocoo()

Dx = sparse.coo_matrix((Dx.ravel(), (IDx.ravel(), JDx.ravel())))
Dx = Dx.tocsr()
Dx = Dx.tocoo()

Dy = sparse.coo_matrix((Dy.ravel(), (IDy.ravel(), JDy.ravel())))
Dy = Dy.tocsr()
Dy = Dy.tocoo()


# Locations of the Wannier boundary 
tol = 1.e-12
tol2 = 1.e-6
#Dflag1 = np.array((abs(np.power(X,2.)+np.power(Y-2.,2.)-1.) < tol))
Dflag2 = np.logical_or.reduce((abs(Y+4*X ) < tol2,
                               abs(Y-4*X) < tol2,
                               abs(Y-2.) < tol2))

ID = np.where(Dflag2)[0]
IDc = np.where(Dflag2==False)[0]

print(ID.size,IDc.size)


freenv = IDc.size
R = sparse.coo_matrix((np.ones(freenv),(np.arange(freenv), IDc )))
R = R.tocsr()
R = R.tocoo()
R_trans = R.transpose()

ux = np.zeros(nv)
uy = np.zeros(nv)
ux[ID] = uex[ID]
uy[ID] = vex[ID]


A = (R.dot(Abar)).dot(R_trans)
A = A.tocsc()
A_inv = sla.inv(A)
DxR = Dx.dot(R_trans)
DyR = Dy.dot(R_trans)

Sx = -Dx.dot(R_trans).dot(A_inv).dot(R).dot(Dx.transpose())
Sy = -Dy.dot(R_trans).dot(A_inv).dot(R).dot(Dy.transpose())

S = Sx + Sy

gp = Dx.dot(ux) + Dy.dot(uy)
fu = -Abar.dot(ux)
fv = -Abar.dot(uy)
fp = gp + Dx.dot(R_trans.dot(A_inv).dot(R.dot(fu))) + Dy.dot(R_trans.dot(A_inv).dot(R.dot(fv)))

p=np.linalg.pinv(S.toarray()).dot(fp)
#p = pe(X,Y)
u = R_trans.dot(A_inv.dot(R.dot(fu)+R.dot(Dx.transpose().dot(p)))) + ux
v = R_trans.dot(A_inv.dot(R.dot(fv)+R.dot(Dy.transpose().dot(p)))) + uy


#Plot:

fig = plt.figure(figsize=(12,12))
triang = tri.Triangulation(X,Y)
triang1= tri.Triangulation(X1,Y1)
ax = fig.add_subplot(3,1,1)
surf = ax.tripcolor(X, Y, u, triangles=E[:,:3], cmap=plt.cm.viridis, linewidth=0.2)
ax.tricontour(triang, u, colors='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('u')
fig.colorbar(surf)
fig.tight_layout()
ax = fig.add_subplot(3,1,2)
surf = ax.tripcolor(X, Y, v, triangles=E[:,:3], cmap=plt.cm.viridis, linewidth=0.2)
ax.tricontour(triang, v, colors='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('v')
fig.colorbar(surf)
fig.tight_layout()
ax = fig.add_subplot(3,1,3)
surf = ax.tripcolor(X1, Y1, p,triangles=E2, cmap=plt.cm.viridis, linewidth=0.2)
ax.tricontour(triang1, p, colors='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('v')
fig.colorbar(surf)
fig.tight_layout()


#Corner Point Pressure:
pp=LinearNDInterpolator(np.vstack([X1,Y1]).T,p)
p_left=pp(np.vstack([-0.50793529,2]).T)
p_right=pp(np.vstack([0.50793529,2]).T)
print('corner point',p_left,p_right)

#Corner Pressure Plot:
#pleft_lst=[-67.99287768,-57.08526459,-51.76269626]
#pright_lst=[ 67.99764429,57.10239,51.76321534]




