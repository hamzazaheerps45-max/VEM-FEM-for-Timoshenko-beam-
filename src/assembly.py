"""
Assembly and solution functions for FEM/VEM analysis.
"""

import numpy as np
import sympy as sp
from .utils import dof_linear, dof_quadratic, dof_mixed

# Import symbolic variables from elements



def assemble_global_system( A, n_elem,dof_conn):
    """
    Assemble global stiffness matrix and load vector for a 1D Timoshenko beam.
    """
    K_e , Q_e , M_e = A
    Q_e = np.array(Q_e).flatten()
    M_e = np.array(M_e).flatten()
    n_dofs = int(np.max(dof_conn) + 1)

    # Allocate global matrices
    K = np.zeros((n_dofs, n_dofs))
    F = np.zeros(n_dofs)

    # Loop over elements
    for e in range(n_elem):
        # global DOF indices for this element
        dofs = dof_conn[e]     # length 4

        # assembly:
        for i_local, I in enumerate(dofs):
            F[I] += Q_e[i_local]
            F[I] += M_e[i_local]
            for j_local, J in enumerate(dofs):
                K[I, J] += K_e[i_local, j_local]

    return K, F




def apply_dirichlet_bc(K, f, dof, value):
    """
    Apply Dirichlet BC: u[dof] = value.
    Modifies K and f directly.
    """


    # Zero the row and column
    K[dof, :] = 0.0
    K[:, dof] = 0.0

    # Put 1 on diagonal
    K[dof, dof] = 1.0

    # Set load vector to the imposed value
    f[dof] = value

def apply_point_load(f, dof, P):
    """
    Add a point load P to DOF index 'dof'
    """
    f[dof] += P

def solve_system(K, F, dirichlet_bcs, point_loads):
    """
    K, f generated from assembly.
    dirichlet_bcs = [(dof, value), ...]
    point_loads    = [(dof, P), ...]
    """

    # Make copies so original K,f are preserved
    K_mod = K.copy()
    F_mod = F.copy()

    # Apply point loads
    for dof, P in point_loads:
        apply_point_load(F_mod, dof, P)

    # Store external load before Dirichlet modification
    Fext_original = F_mod.copy()

    # Apply Dirichlet BCs
    for dof, value in dirichlet_bcs:
        apply_dirichlet_bc(K_mod, F_mod, dof, value)

    # Solve the system
    U = np.linalg.solve(K_mod, F_mod)

    # Compute reactions: r = K*u - fext
    Fr = K @ U - Fext_original

    # Total force vector f_total = fext + fr
    F_total = Fext_original + Fr

    return U, F_total


def Timosheko_beam( Input_data ,Boundary_parameter):
    L,EI,GA, no_ele,Method = Input_data
    dirichlet_bcs , point_loads , q_e , m_e = Boundary_parameter
    K_func,Q_func,M_func , order= Method()

    le = L/no_ele
    
 
    K_e =  K_func(le,EI,GA)
    Q_e = Q_func(le,EI,GA,q_e)
    M_e= M_func(le,EI,GA,m_e )
    A= [K_e,Q_e,M_e]

    if order == "Linear":
        dof_conn = dof_linear(no_ele)
    elif order == "Quadratic":
        dof_conn = dof_quadratic(no_ele)
    elif order == "Mixed":
        dof_conn = dof_mixed(no_ele)
    else:
        print("Error")

    K , F = assemble_global_system( A, no_ele,dof_conn)
    U, F = solve_system(K, F, dirichlet_bcs, point_loads)
    return U,F

