"""src.elements
=================================
Timoshenko beam element formulations (FEM and VEM).

This module provides symbolic element routines using :mod:`sympy` that
construct element stiffness matrices, body-force vectors and moment
vectors for different interpolation orders. Each element factory
returns numeric lambdified callables for the matrix/vector that accept
material/geometry parameters.
"""


import numpy as np
import sympy as sp




# Linear transverse displacement (w) and linear rotation (phi).
# Degrees of freedom: u = [w1, φ1, w2, φ2]
def FEM_11():
    x , xi , EI ,GA ,le, x1,x2  ,q , m , = sp.symbols('x xi EI GA le x1 x2  q m ')
    # Shape functions and their derivatives for linear FEM
    N1 = (1 - xi)/2
    N2 = (1 + xi)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)

    # Isoparametric mapping
    x = sp.Matrix([N1, N2]).T * sp.Matrix([x1, x2])
    # Jacobian
    J = sp.diff(x,xi)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x2 - x1,le)
    detJ = J
    J_inv = 1/J
    # Material constitutive matrix
    C = sp.Matrix([[EI,0],[0,GA]])
    

    # stiffness Matrix
    B = sp.Matrix([[0,J_inv*dN1_dξ,0,J_inv*dN2_dξ],[J_inv*dN1_dξ,-N1,J_inv*dN2_dξ,-N2]])
    K = B.T*C*B *detJ
    K= K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)



    # Body force and distributed moment
    Q = sp.Matrix([N1,0,N2,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    M = sp.Matrix([0,N1,0,N2])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func ,"Linear"





  

# Linear w and linear phi using reduced integration for shear terms
def FEM_11_red():
    x , xi , EI ,GA ,le, x1,x2  ,q , m  = sp.symbols('x xi EI GA le x1 x2  q m ')
    # Shape functions and their derivatives for linear FEM
    N1 = (1 - xi)/2
    N2 = (1 + xi)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)
    N1_red= N1.subs(xi , 0)
    N2_red= N2.subs(xi , 0)
    # Isoparametric mapping
    x = sp.Matrix([N1 , N2]).T*sp.Matrix([x1 , x2])
    #Jacobian 
    J = sp.diff(x,xi)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x2 - x1,le)
    detJ = J
    J_inv= 1/J
    # Material constitutive matrix
    C = sp.Matrix([[EI,0],[0,GA]])


    # Stiffness matrix (reduced integration applied only to shear part)
    B = sp.Matrix([[0,J_inv*dN1_dξ,0,J_inv*dN2_dξ],[J_inv*dN1_dξ,-N1_red,J_inv*dN2_dξ,-N2_red]])
    K = B.T*C*B*detJ
    K= K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)



    # Body force and distributed moment
    Q = sp.Matrix([N1,0,N2,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    M = sp.Matrix([0,N1,0,N2])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func ,"Linear"





  


# Quadratic w and quadratic phi.
# Degrees of freedom: u = [w1, φ1, w2, φ2, w3, φ3]
def FEM_22():
    x , xi , EI ,GA ,le, x1,x2 ,x3 ,q , m  = sp.symbols('x xi EI GA le x1 x2 x3 q m ')
    # Shape functions and their derivatives for quadratic FEM
    N1 = xi*(xi - 1)/2
    N2 = 1 - xi**2
    N3 = xi*(xi + 1)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)
    dN3_dξ = sp.diff(N3,xi)
    # Isoparametric mapping (quadratic)
    x = sp.Matrix([N1 , N2 , N3]).T*sp.Matrix([x1 , x2 ,x3])
    #Jacobian 
    J = sp.diff(x,xi)
    J = J.subs(x2, (x1+x3)/2)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x3 - x1,le)
    detJ = J
    J_inv = 1/J

    # Material constitutive matrix
    C = sp.Matrix([[EI,0],[0,GA]])

    # stiffness Matrix
    B = sp.Matrix([[0,J_inv*dN1_dξ,0,J_inv*dN2_dξ,0,J_inv*dN3_dξ],
                    [J_inv*dN1_dξ,-N1,J_inv*dN2_dξ,-N2,J_inv*dN3_dξ,-N3]])
    K = B.T*C*B *detJ
    K = K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)

    # Body force and distributed moment
    Q = sp.Matrix([N1,0,N2,0,N3,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    
    M = sp.Matrix([0,N1,0,N2,0,N3])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func ,"Quadratic"



# Quadratic w and linear phi (mixed interpolation).
# DOFs: u = [w1, φ1, w2, w3, φ3] (φ at middle node omitted)
def FEM_21():
    x , xi , EI ,GA ,le, x1,x2 ,x3 ,q , m  = sp.symbols('x xi EI GA le x1 x2 x3 q m ')
    # Shape functions and derivatives for linear rotation φ
    M1 = (1 - xi)/2
    M2 = (1 + xi)/2
    dM1_dξ = sp.diff(M1,xi)
    dM2_dξ = sp.diff(M2,xi)
    # Shape functions and derivatives for quadratic w interpolation
    N1 = xi*(xi - 1)/2
    N2 = 1 - xi**2
    N3 = xi*(xi + 1)/2
    dN1_dξ = sp.diff(N1,xi)
    dN2_dξ = sp.diff(N2,xi)
    dN3_dξ = sp.diff(N3,xi)
    # Isoparametric mapping (quadratic mapping)
    x = sp.Matrix([N1 , N2 , N3]).T*sp.Matrix([x1 , x2 ,x3])
    #Jacobian 
    J = sp.diff(x,xi)
    J = J.subs(x2, (x1+x3)/2)
    J = sp.simplify(J[0,0])
    J= sp.factor(J)
    J = J.subs(x3 - x1,le)
    detJ = J
    J_inv = 1/J
    # Material constitutive matrix
    C = sp.Matrix([[EI,0],[0,GA]])

    # Stiffness matrix (linear φ, quadratic w)

    B = sp.Matrix([[0,J_inv*dM1_dξ,0,0,J_inv*dM2_dξ],
                        [J_inv*dN1_dξ,-M1,J_inv*dN2_dξ,J_inv*dN3_dξ,-M2]])
    K = B.T*C*B *detJ
    K = K.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K = K.applyfunc(sp.factor)

    # Body force and distributed moment
    Q = sp.Matrix([N1,0,N2,N3,0])*q*detJ
    Q = Q.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
  
    M = sp.Matrix([0,M1,0,0,M2])*m*detJ
    M = M.applyfunc(lambda f: sp.integrate(f, (xi, -1, 1)))
    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func , "Mixed"


# Virtual Element Method — linear w and linear φ (no internal moment)
def VEM_11():
    x  , EI ,GA ,le ,q , m , w1,w2  = sp.symbols('x  EI GA le q m w1 w2 ') 
    N_p = sp.Matrix([1, x])
    B_p = sp.Matrix([1])
    G = B_p*B_p.T
    G = G.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    r = sp.Matrix([w2-w1])-sp.Matrix([0])
    G_inv = G.inv()
    a_hat = G_inv*r
    dof = sp.Matrix([w1,w2])
    B_bar = a_hat.jacobian(dof)
    P = sp.Matrix([[1,0],B_bar[0,:]])

    # Shape functions for the Virtual Element Method
    N = N_p.T*P
    N_bar= B_bar.T*B_p

    # Material constitutive matrix
    C = sp.Matrix([[EI,0],[0,GA]])

    B = sp.Matrix([[0,N_bar[0],0,N_bar[1]],[N_bar[0],-N[0],N_bar[1],-N[1]]])
    K = B.T*C*B
    K = K.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K = K.applyfunc(sp.factor)


# body Force and Moment
    Q = sp.Matrix([N[0],0,N[1],0])*q
    Q = Q.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
 
    M = sp.Matrix([0,N[0],0,N[1]])*m
    M = M.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func ,"Linear"



# VEM Quaratic W and Quaratic theta u =[w1,φ1,w2,φ2,w3,φ3] (w2,φ2 is internal moment) 

def VEM_22():
    x, EI, GA, le, q, m, w1, w2, m0 = sp.symbols('x EI GA le q m w1 w2 m0')
    N_p = sp.Matrix([1, x, x**2])
    B_p = sp.Matrix([1, 2*x])
    G = B_p*B_p.T
    G = G.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    r = sp.Matrix([w2-w1,2*le*w2])-sp.Matrix([0,2*le*m0])
    G_inv = G.inv()
    a_hat = G_inv*r
    dof = sp.Matrix([w1,w2,m0])
    B_bar = a_hat.jacobian(dof)
    P = sp.Matrix([[1,0,0],B_bar[0,:],B_bar[1,:]])

    # Shape functions for the Virtual Element Method

    N = N_p.T*P
    N_bar= B_bar.T*B_p

# Material Modulus
    C = sp.Matrix([[EI,0],[0,GA]])

    B = sp.Matrix([[0,N_bar[0],0,N_bar[2],0,N_bar[1]],[N_bar[0],-N[0],N_bar[2],-N[2],N_bar[1],-N[1]]])
    K = B.T*C*B
    K = K.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K = K.applyfunc(sp.factor)


    # Body force and distributed moment
    Q = sp.Matrix([N[0],0,N[2],0,N[1],0])*q
    Q = Q.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
 
    M = sp.Matrix([0,N[0],0,N[2],0,N[1]])*m
    M = M.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func ,"Quadratic"




# VEM quadratic w and linear φ (mixed virtual interpolation)
# DOFs: u = [w1, φ1, w2, w3, φ3] (No internal moment for φ)
def VEM_21():
    x, xi, EI, GA, le, x1, x2, x3, q, m, w1, w2, m0, m1 = sp.symbols(
        'x xi EI GA le x1 x2 x3 q m w1 w2 m0 m1')
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

    # Shape functions for Virtual Element Method (linear part)
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

    # Shape functions for Virtual Element Method (quadratic part)
    N = N_p.T*P
    N_bar= B_bar.T*B_p
    
    C = sp.Matrix([[EI,0],[0,GA]])

    B = sp.Matrix([[0,A_bar[0],0,0,A_bar[1]],
                        [N_bar[0],-A[0],N_bar[2],N_bar[1],-A[1]]])
    K = B.T*C*B
    K= K.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    K= K.applyfunc(sp.factor)

    # Body force and distributed moment
    Q = sp.Matrix([N[0],0,N[2],N[1],0])*q
    Q = Q.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))
    M = sp.Matrix([0,A[0],0,0,A[1]])*m
    M = M.applyfunc(lambda f: sp.integrate(f, (x, 0, le)))

    K_func= sp.lambdify((le,EI,GA ), K, "numpy")
    Q_func= sp.lambdify((le,EI,GA,q ), Q, "numpy")
    M_func= sp.lambdify((le,EI,GA,m ), M, "numpy")

    return K_func , Q_func , M_func ,"Mixed"

# Linear shape function evaluation for VEM elements
def VEM_Shape_function_Linear():
    x, le, w1, w2 = sp.symbols('x le w1 w2')
    N_p = sp.Matrix([1, x])
    B_p = sp.Matrix([1])
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
    N_func = sp.lambdify((x,le),N, "numpy")
    N_bar_func = sp.lambdify((x,le),N_bar, "numpy")
    return N_func , N_bar_func


# Quadratic shape function evaluation for VEM elements
def VEM_Shape_function_Quadratic():
    x, le, w1, w2, m0 = sp.symbols('x le w1 w2 m0')
    N_p = sp.Matrix([1, x, x**2])
    B_p = sp.Matrix([1, 2*x])
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
    N_func = sp.lambdify((x,le),N, "numpy")
    N_bar_func = sp.lambdify((x,le),N_bar, "numpy")
    return N_func , N_bar_func


