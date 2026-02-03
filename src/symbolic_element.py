"""
Timoshenko Beam Element symbolic Formulations
=====================================
This module contains symbolic FEM and VEM element formulations for Timoshenko beams.
"""


import numpy as np
import sympy as sp
from sympy import init_printing, latex, symbols, Matrix


# Define symbolic variables
x , xi , EI ,GA ,le, x1,x2 ,x3 ,q , m , w1,w2, m0 = sp.symbols('x xi EI GA le x1 x2 x3 q m w1 w2 m0')


# Linear W and Linear theta u =[w1,φ1,w2,φ2]
def FEM_11():
    # Shape and their Derivative fucntion for linear FEM
    N1 = (1 - xi)/2
    N2 = (1 + xi)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)

    # isoparametric mapping
    x = sp.Matrix([N1 , N2]).T*sp.Matrix([x1 , x2])
    #Jacobian 
    J = sp.diff(x,xi)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x2 - x1,le)
    detJ = J
    J_inv = 1/J
    # Material Modulus
    C = sp.Matrix([[EI,0],[0,GA]])
    

    # stiffness Matrix
    B = sp.Matrix([[0,J_inv*dN1_dξ,0,J_inv*dN2_dξ],[J_inv*dN1_dξ,-N1,J_inv*dN2_dξ,-N2]])
    K = B.T*C*B *detJ
    K= K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)



    # body Force and Moment 
    Q = sp.Matrix([N1,0,N2,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    M = sp.Matrix([0,N1,0,N2])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))


    return K , Q , M 


# Linear W and Linear theta u =[w1,φ1,w2,φ2] (using reduced integration for shearing part)
def FEM_11_red():
    # Shape and their Derivative fucntion for linear FEM
    N1 = (1 - xi)/2
    N2 = (1 + xi)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)
    N1_red= N1.subs(xi , 0)
    N2_red= N2.subs(xi , 0)
    # isoparametric mapping
    x = sp.Matrix([N1 , N2]).T*sp.Matrix([x1 , x2])
    #Jacobian 
    J = sp.diff(x,xi)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x2 - x1,le)
    detJ = J
    J_inv= 1/J
    # Material Modulus
    C = sp.Matrix([[EI,0],[0,GA]])


    # stiffness Matrix (reduced integration only on shearing part)
    B = sp.Matrix([[0,J_inv*dN1_dξ,0,J_inv*dN2_dξ],[J_inv*dN1_dξ,-N1_red,J_inv*dN2_dξ,-N2_red]])
    K = B.T*C*B*detJ
    K= K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)



    # body Force and Moment 
    Q = sp.Matrix([N1,0,N2,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    M = sp.Matrix([0,N1,0,N2])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))


    return K , Q , M 




# Quadratic W and Quadratic theta u =[w1,φ1,w2,φ2,w3,φ3] 
def FEM_22():
    # Shape and their Derivative fucntion for quadratic FEM
    N1 = xi*(xi - 1)/2
    N2 = 1 - xi**2
    N3 = xi*(xi + 1)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)
    dN3_dξ = sp.diff(N3,xi)
    # isoparametric mapping
    x = sp.Matrix([N1 , N2 , N3]).T*sp.Matrix([x1 , x2 ,x3])
    #Jacobian 
    J = sp.diff(x,xi)
    J = J.subs(x2, (x1+x3)/2)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x3 - x1,le)
    detJ = J
    J_inv = 1/J

    # Material Modulus
    C = sp.Matrix([[EI,0],[0,GA]])

    # stiffness Matrix
    B = sp.Matrix([[0,J_inv*dN1_dξ,0,J_inv*dN2_dξ,0,J_inv*dN3_dξ],
                    [J_inv*dN1_dξ,-N1,J_inv*dN2_dξ,-N2,J_inv*dN3_dξ,-N3]])
    K = B.T*C*B *detJ
    K = K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)

    # body Force and Moment 
    Q = sp.Matrix([N1,0,N2,0,N3,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    
    M = sp.Matrix([0,N1,0,N2,0,N3])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    return K , Q , M 



# Quadratic W and Quadratic theta u =[w1,φ1,w3,φ3] (removing middle node)
def FEM_22_cond():
    K,Q,M = FEM_22()
    # Static condensation 
    K_rr = K[[0,1,4,5], [0,1,4,5]]
    K_rc = K[[0,1,4,5], [2,3]]
    K_cr = K[[2,3], [0,1,4,5]]
    K_cc = K[[2,3],[2,3]]

    K_cond= K_rr - K_rc * (K_cc.inv()) * K_cr
    K_cond= K_cond.applyfunc(sp.factor)

    Q_r = Q[[0,1,4,5],0]
    Q_c = Q[[2,3],0]
    Q_cond = Q_r - K_rc * (K_cc.inv()) * Q_c


    M_r = M[[0,1,4,5],0]
    M_c = M[[2,3],0]
    M_cond = M_r - K_rc * (K_cc.inv()) * M_c
    M_cond = sp.simplify(M_cond)
    return K_cond ,Q_cond,M_cond


# Quadratic W and linear theta u =[w1,φ1,w2,w3,φ3] (no φ2, w2 is at middle node )
def FEM_21():
    # Shape and their Derivative fucntion for φ(lin)
    M1 = (1 - xi)/2
    M2 = (1 + xi)/2
    dM1_dξ = sp.diff(M1,xi)
    dM2_dξ = sp.diff(M2,xi)
     # Shape and their Derivative fucntion for quadratic FEM
    N1 = xi*(xi - 1)/2
    N2 = 1 - xi**2
    N3 = xi*(xi + 1)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)
    dN3_dξ = sp.diff(N3,xi)
    # isoparametric mapping (using higher order(2) )
    x = sp.Matrix([N1 , N2 , N3]).T*sp.Matrix([x1 , x2 ,x3])
    #Jacobian 
    J = sp.diff(x,xi)
    J = J.subs(x2, (x1+x3)/2)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x3 - x1,le)
    detJ = J
    J_inv = 1/J
    # Material Modulus
    C = sp.Matrix([[EI,0],[0,GA]])

    # stiffness Matrix(using linear for φ and quadratic for w)

    B = sp.Matrix([[0,J_inv*dM1_dξ,0,0,J_inv*dM2_dξ],
                        [J_inv*dN1_dξ,-M1,J_inv*dN2_dξ,J_inv*dN3_dξ,-M2]])
    K = B.T*C*B *detJ
    K = K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)

    # Body Force and Moment 
    Q = sp.Matrix([N1,0,N2,N3,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
  
    M = sp.Matrix([0,M1,0,0,M2])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))

    return K , Q , M


    
# Quadratic W and linear theta u =[w1,φ1,w3,φ3] (removing w2 )
def FEM_21_cond():
    K,Q,M = FEM_21()
    # Static condensation 
    K_rr = K[[0,1,3,4], [0,1,3,4]]
    K_rc = K[[0,1,3,4], [2]]
    K_cr = K[[2], [0,1,3,4]]
    K_cc = K[2,2]
    K_cond= K_rr - K_rc * (1/K_cc) * K_cr
    K_cond= K_cond.applyfunc(sp.factor)
    Q_r = Q[[0,1,3,4],0]

    Q_c = Q[2]
    Q_cond = Q_r - K_rc * (1/K_cc) * Q_c
    M_cond = M[[0,1,3,4],0]

    return K_cond ,Q_cond,M_cond


def VEM_11(): 
    N_p = sp.Matrix([1 ,x ])
    B_p = sp.Matrix([1 ])
    G = B_p*B_p.T
    G = G.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    r = sp.Matrix([w2-w1])-sp.Matrix([0])
    G_inv = G.inv()
    a_hat = G_inv*r
    dof = sp.Matrix([w1,w2])
    B_bar = a_hat.jacobian(dof)
    P = sp.Matrix([[1,0],B_bar[0,:]])


    N = N_p.T*P
    N_bar= B_bar.T*B_p


    C = sp.Matrix([[EI,0],[0,GA]])

    B = sp.Matrix([[0,N_bar[0],0,N_bar[1]],[N_bar[0],-N[0],N_bar[1],-N[1]]])
    K = B.T*C*B
    K = K.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K = K.applyfunc(sp.factor)



    Q = sp.Matrix([N[0],0,N[1],0])*q
    Q = Q.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
 
    M = sp.Matrix([0,N[0],0,N[1]])*m
    M = M.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    return K,Q,M





def VEM_22(): 
    N_p = sp.Matrix([1 ,x , x**2])
    B_p = sp.Matrix([1 ,2*x ])
    G = B_p*B_p.T
    G = G.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    r = sp.Matrix([w2-w1,2*le*w2])-sp.Matrix([0,2*le*m0])
    G_inv = G.inv()
    a_hat = G_inv*r
    dof = sp.Matrix([w1,w2,m0])
    B_bar = a_hat.jacobian(dof)
    P = sp.Matrix([[1,0,0],B_bar[0,:],B_bar[1,:]])


    N = N_p.T*P
    N_bar= B_bar.T*B_p


    C = sp.Matrix([[EI,0],[0,GA]])

    B = sp.Matrix([[0,N_bar[0],0,N_bar[2],0,N_bar[1]],[N_bar[0],-N[0],N_bar[2],-N[2],N_bar[1],-N[1]]])
    K = B.T*C*B
    K = K.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K = K.applyfunc(sp.factor)



    Q = sp.Matrix([N[0],0,N[2],0,N[1],0])*q
    Q = Q.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
 
    M = sp.Matrix([0,N[0],0,N[2],0,N[1]])*m
    M = M.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    return K,Q,M




def VEM_22_cond():
    K,Q,M =VEM_22()
    K_rr = K[[0,1,4,5], [0,1,4,5]]
    K_rc = K[[0,1,4,5], [2,3]]
    K_cr = K[[2,3], [0,1,4,5]]
    K_cc = K[[2,3],[2,3]]

    K_cond= K_rr - K_rc * (K_cc.inv()) * K_cr
    K_cond= K_cond.applyfunc(sp.factor)
    Q_r = Q[[0,1,4,5],0]

    Q_c = Q[[2,3],0]
    Q_cond = Q_r - K_rc * (K_cc.inv()) * Q_c

    

    M_r = M[[0,1,4,5],0]

    M_c = M[[2,3],0]
    M_cond = M_r - K_rc * (K_cc.inv()) * M_c
    M_cond= M_cond.applyfunc(sp.factor)

    return  K_cond ,Q_cond,M_cond


def VEM_21():   
    N_p = sp.Matrix([1 ,x ])
    B_p = sp.Matrix([1 ])
    G = B_p*B_p.T
    G = G.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    r = sp.Matrix([w2-w1])-sp.Matrix([0])
    G_inv = G.inv()
    a_hat = G_inv*r
    dof = sp.Matrix([w1,w2])
    B_bar = a_hat.jacobian(dof)
    P = sp.Matrix([[1,0],B_bar[0,:]])


    A = N_p.T*P
    A_bar= B_bar.T*B_p

    N_p = sp.Matrix([1 ,x , x**2])
    B_p = sp.Matrix([1 ,2*x ])
    G = B_p*B_p.T
    G = G.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    r = sp.Matrix([w2-w1,2*le*w2])-sp.Matrix([0,2*le*m0])
    G_inv = G.inv()
    a_hat = G_inv*r
    dof = sp.Matrix([w1,w2,m0])
    B_bar = a_hat.jacobian(dof)
    P = sp.Matrix([[1,0,0],B_bar[0,:],B_bar[1,:]])


    N = N_p.T*P
    N_bar= B_bar.T*B_p
    
    C = sp.Matrix([[EI,0],[0,GA]])

    B = sp.Matrix([[0,A_bar[0],0,0,A_bar[1]],
                        [N_bar[0],-A[0],N_bar[2],N_bar[1],-A[1]]])
    K = B.T*C*B
    K= K.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K= K.applyfunc(sp.factor)

    
    Q = sp.Matrix([N[0],0,N[2],N[1],0])*q
    Q = Q.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    M = sp.Matrix([0,A[0],0,0,A[1]])*m
    M = M.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))



    return K , Q ,M


def VEM_21_cond():
    K,Q,M = VEM_21()
    # Static condensation 
    K_rr = K[[0,1,3,4], [0,1,3,4]]
    K_rc = K[[0,1,3,4], [2]]
    K_cr = K[[2], [0,1,3,4]]
    K_cc = K[2,2]
    K_cond= K_rr - K_rc * (1/K_cc) * K_cr
    K_cond= K_cond.applyfunc(sp.factor)
    Q_r = Q[[0,1,3,4],0]

    Q_c = Q[2]
    Q_cond = Q_r - K_rc * (1/K_cc) * Q_c
    M_cond = M[[0,1,3,4],0]
    M_cond= M_cond.applyfunc(sp.factor)
    return K_cond ,Q_cond,M_cond
