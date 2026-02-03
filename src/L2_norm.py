import numpy as np
import sympy as sp
import math
from .utils import dof_linear, dof_quadratic, dof_mixed ,node_coords_linear ,node_coords_quadratic
from.elements import VEM_Shape_function_Linear,VEM_Shape_function_Quadratic






# FEM Linear L2 norm computation

def compute_L2_and_relative_errors_FL(U,Input_data):


    L,EI,GA, no_ele,no_guass_point,exact_funcs = Input_data
    le = L / no_ele

   

    # Gauss weights and points
    xi_points,Wp= np.polynomial.legendre.leggauss(no_guass_point)

    # Connectivity
    dof_conn = dof_linear(no_ele)
    node_coords = node_coords_linear(no_ele, L)

    # Initialize L2 accumulators
    L2_w = 0.0
    L2_th = 0.0
    L2_M = 0.0
    L2_V = 0.0

    L2_w_exact = 0.0
    L2_th_exact = 0.0
    L2_M_exact = 0.0
    L2_V_exact = 0.0

         

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2]
        dofs = dof_conn[e]
        u_e = U[dofs]                   # [W1, φ1, W2, φ2]

        W1, φ1, W2, φ2 = u_e
        J = le / 2 
        J_inv = 1/J
        # shape function derivatives dN/dx
        dN1_dx = -0.5*J_inv
        dN2_dx =  0.5*J_inv

        # shape functions
        def N1(ξ): return 0.5 * (1 - ξ)
        def N2(ξ): return 0.5 * (1 + ξ)

        for gp in range(no_guass_point):
            ξ = xi_points[gp]
            w_g = Wp[gp]

            # shape values
            N1ξ = N1(ξ)
            N2ξ = N2(ξ)

            # FEM fields
            W = N1ξ*W1 + N2ξ*W2
            φ = N1ξ*φ1 + N2ξ*φ2
            dw_dx = dN1_dx*W1 + dN2_dx*W2
            dφ_dx = dN1_dx*φ1 + dN2_dx*φ2

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)

            # physical coordinate
            x_phys = N1ξ*x_local[0] + N2ξ*x_local[1]
            
            exact_value = exact_funcs(x_phys)
           
            # exact fields
            exact_value = np.asarray(exact_funcs(x_phys)).flatten()

            w_exact  = float(exact_value[0])
            th_exact = float(exact_value[1])
            M_exact  = float(exact_value[2])
            V_exact  = float(exact_value[3])


            # accumulate L2 errors
            L2_w  += w_g * J * (w_exact  - W)**2
            L2_th += w_g * J * (th_exact - φ)**2
            L2_M  += w_g * J * (M_exact  - M)**2
            L2_V  += w_g * J * (V_exact  - V)**2

            # accumulate exact norms
            L2_w_exact  += w_g * J * (w_exact )**2
            L2_th_exact += w_g * J * (th_exact)**2
            L2_M_exact  += w_g * J * (M_exact )**2
            L2_V_exact  += w_g * J * (V_exact )**2

    # Absolute L2 errors
    abs_L2 = (
       np.sqrt( L2_w),
        np.sqrt( L2_th),
        np.sqrt( L2_M),
        np.sqrt( L2_V),
    )

    # Relative errors
    rel_L2 = (
        np.sqrt(L2_w / L2_w_exact)   if L2_w_exact  > 0 else 0,
        np.sqrt(L2_th / L2_th_exact) if L2_th_exact > 0 else 0,
        np.sqrt(L2_M / L2_M_exact)   if L2_M_exact  > 0 else 0,
        np.sqrt(L2_V / L2_V_exact)   if L2_V_exact  > 0 else 0,
    )

    return abs_L2, rel_L2

# FEM Quadratic L2 norm computation
def compute_L2_and_relative_errors_FQ(U, Input_data):

    import numpy as np
    L,EI,GA, no_ele,no_guass_point,exact_funcs = Input_data
 
    le = L / no_ele

    

    # Gauss weights and points
    xi_points,Wp = np.polynomial.legendre.leggauss(no_guass_point)

    # Connectivity
    dof_conn = dof_quadratic(no_ele)
    node_coords = node_coords_quadratic(no_ele, L)

    # Initialize L2 accumulators
    L2_w = 0.0
    L2_th = 0.0
    L2_M = 0.0
    L2_V = 0.0

    L2_w_exact = 0.0
    L2_th_exact = 0.0
    L2_M_exact = 0.0
    L2_V_exact = 0.0

         

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2]
        dofs = dof_conn[e]
        u_e = U[dofs]                   # [W1, φ1, W2, φ2]

        x_local = node_coords[e]        # [x1, x2,x3]
        dofs = dof_conn[e]
        u_e = U[dofs]                  

        W1, φ1, W2, φ2, W3, φ3 = u_e

        # shape function derivatives
        J = le/2
        J_inv = 1/J

      

        # shape functions
        def N1(ξ): return ξ*(ξ-1)/2
        def N2(ξ): return (1-ξ**2)
        def N3(ξ): return ξ*(ξ+1)/2
        def dN1dx(ξ): return J_inv*(2*ξ-1)/2
        def dN2dx(ξ): return J_inv*(-2*ξ)
        def dN3dx(ξ): return J_inv*(2*ξ+1)/2

        for gp in range(no_guass_point):
            ξ = xi_points[gp]
            w_g = Wp[gp]

            N1ξ = N1(ξ)
            N2ξ = N2(ξ)
            N3ξ = N3(ξ)
            dN1_dx = dN1dx(ξ)
            dN2_dx = dN2dx(ξ)
            dN3_dx = dN3dx(ξ)
            W = N1ξ*W1 + N2ξ* W2+ N3ξ*W3
            φ = N1ξ*φ1 + N2ξ*φ2 + N3ξ*φ3

            dw_dx = dN1_dx*W1 + dN2_dx*W2+dN3_dx*W3
            dφ_dx = dN1_dx*φ1 + dN2_dx*φ2 + dN3_dx*φ3

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)

            x_phys = N1ξ*x_local[0] + N2ξ*x_local[1]+N3ξ*x_local[2]


            exact_value = exact_funcs(x_phys)
            # exact fields
            exact_value = np.asarray(exact_funcs(x_phys)).flatten()

            w_exact  = float(exact_value[0])
            th_exact = float(exact_value[1])
            M_exact  = float(exact_value[2])
            V_exact  = float(exact_value[3])

            # accumulate L2 errors
            L2_w  += w_g * J * (w_exact  - W)**2
            L2_th += w_g * J * (th_exact - φ)**2
            L2_M  += w_g * J * (M_exact  - M)**2
            L2_V  += w_g * J * (V_exact  - V)**2

            # accumulate exact norms
            L2_w_exact  += w_g * J * (w_exact )**2
            L2_th_exact += w_g * J * (th_exact)**2
            L2_M_exact  += w_g * J * (M_exact )**2
            L2_V_exact  += w_g * J * (V_exact )**2

    # Absolute L2 errors
    abs_L2 = (
        np.sqrt(L2_w),
        np.sqrt(L2_th),
        np.sqrt(L2_M),
        np.sqrt(L2_V),
    )

    # Relative errors
    rel_L2 = (
        np.sqrt(L2_w / L2_w_exact)   if L2_w_exact  > 0 else 0,
        np.sqrt(L2_th / L2_th_exact) if L2_th_exact > 0 else 0,
        np.sqrt(L2_M / L2_M_exact)   if L2_M_exact  > 0 else 0,
        np.sqrt(L2_V / L2_V_exact)   if L2_V_exact  > 0 else 0,
    )

    return abs_L2, rel_L2




# FEM Mixed L2 norm computation
def compute_L2_and_relative_errors_FM(U, Input_data):

    L,EI,GA, no_ele,no_guass_point,exact_funcs = Input_data
    le = L / no_ele

  

    # Gauss weights and points
    xi_points,Wp = np.polynomial.legendre.leggauss(no_guass_point)

    # Connectivity
    dof_conn = dof_mixed(no_ele)
    node_coords = node_coords_quadratic(no_ele, L)

    # Initialize L2 accumulators
    L2_w = 0.0
    L2_th = 0.0
    L2_M = 0.0
    L2_V = 0.0

    L2_w_exact = 0.0
    L2_th_exact = 0.0
    L2_M_exact = 0.0
    L2_V_exact = 0.0

         

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2,x3]
        dofs = dof_conn[e]
        u_e = U[dofs]                  

        W1, φ1, W2, W3, φ3 = u_e

        # shape function derivatives
        J = le/2
        J_inv = 1 / J

      

        # shape functions
        def N1(ξ): return ξ*(ξ-1)/2
        def N2(ξ): return (1-ξ**2)
        def N3(ξ): return ξ*(ξ+1)/2
        def dN1dx(ξ): return J_inv*(2*ξ-1)/2
        def dN2dx(ξ): return J_inv*(-2*ξ)
        def dN3dx(ξ): return J_inv*(2*ξ+1)/2
        dM1_dξ = -0.5
        dM3_dξ =  0.5
       

        dM1_dx = dM1_dξ * J_inv
        dM3_dx = dM3_dξ * J_inv

        # shape functions
        def M1(ξ): return 0.5*(1-ξ)
        def M3(ξ): return 0.5*(1+ξ)

        for gp in range(no_guass_point):
            ξ = xi_points[gp]
            w_g = Wp[gp]

            N1ξ = N1(ξ)
            N2ξ = N2(ξ)
            N3ξ = N3(ξ)
            M1ξ = M1(ξ)
            M3ξ = M3(ξ)
            dN1_dx = dN1dx(ξ)
            dN2_dx = dN2dx(ξ)
            dN3_dx = dN3dx(ξ)
            W = N1ξ*W1 + N2ξ* W2+ N3ξ*W3
            φ = M1ξ*φ1 + M3ξ*φ3 

            dw_dx = dN1_dx*W1 + dN2_dx*W2+dN3_dx*W3
            dφ_dx = dM1_dx*φ1 + dM3_dx*φ3 

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)

            x_phys = N1ξ*x_local[0] + N2ξ*x_local[1]+N3ξ*x_local[2]



            exact_value = exact_funcs(x_phys)
            # exact fields
            exact_value = np.asarray(exact_funcs(x_phys)).flatten()

            w_exact  = float(exact_value[0])
            th_exact = float(exact_value[1])
            M_exact  = float(exact_value[2])
            V_exact  = float(exact_value[3])

            # accumulate L2 errors
            L2_w  += w_g * J * (w_exact  - W)**2
            L2_th += w_g * J * (th_exact - φ)**2
            L2_M  += w_g * J * (M_exact  - M)**2
            L2_V  += w_g * J * (V_exact  - V)**2

            # accumulate exact norms
            L2_w_exact  += w_g * J * (w_exact )**2
            L2_th_exact += w_g * J * (th_exact)**2
            L2_M_exact  += w_g * J * (M_exact )**2
            L2_V_exact  += w_g * J * (V_exact )**2

    # Absolute L2 errors
    abs_L2 = (
        np.sqrt(L2_w),
        np.sqrt(L2_th),
        np.sqrt(L2_M),
        np.sqrt(L2_V),
    )

    # Relative errors
    rel_L2 = (
        np.sqrt(L2_w / L2_w_exact)   if L2_w_exact  > 0 else 0,
        np.sqrt(L2_th / L2_th_exact) if L2_th_exact > 0 else 0,
        np.sqrt(L2_M / L2_M_exact)   if L2_M_exact  > 0 else 0,
        np.sqrt(L2_V / L2_V_exact)   if L2_V_exact  > 0 else 0,
    )

    return abs_L2, rel_L2





# VEM Mixed L2 norm computation
def compute_L2_and_relative_errors_VM(U, Input_data):

    L,EI,GA, no_ele,no_guass_point,exact_funcs = Input_data
    le = L / no_ele

  

    # Gauss weights and points
    xi_points,Wp = np.polynomial.legendre.leggauss(no_guass_point)

    # Connectivity
    dof_conn = dof_mixed(no_ele)
    node_coords = node_coords_linear(no_ele, L)
    K_func , K_bar_func = VEM_Shape_function_Linear()
    N_func , N_bar_func = VEM_Shape_function_Quadratic()

    # Initialize L2 accumulators
    L2_w = 0.0
    L2_th = 0.0
    L2_M = 0.0
    L2_V = 0.0
    
    L2_w_exact = 0.0
    L2_th_exact = 0.0
    L2_M_exact = 0.0
    L2_V_exact = 0.0

         

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2]
        dofs = dof_conn[e]
        u_e = U[dofs]                  

       
        W1, φ1, W2, W3, φ3 = u_e
        W_vec= np.array([W1,W3,W2])   # in formulation tthe order is W1(left node) , W3(right node) , W2(internal moment)
        φ_vector = np.array([φ1,φ3])
        # shape function derivatives
        J = le/2

      

        for gp in range(no_guass_point):
            ξ = xi_points[gp]
            w_g = Wp[gp]
            x_phys = le/2 * (1 + ξ)

            N = N_func(x_phys,le)
            N_bar = N_bar_func(x_phys,le)
            K = K_func(x_phys,le)
            K_bar=  K_bar_func(x_phys,le)
            
            N     = np.array(N,     dtype=float).reshape(-1)
            N_bar = np.array(N_bar, dtype=float).reshape(-1)
            K     = np.array(K,     dtype=float).reshape(-1)
            K_bar = np.array(K_bar, dtype=float).reshape(-1)

            W = float(np.dot(N, W_vec))
            φ = float(np.dot(K, φ_vector))

            dw_dx = float(np.dot(N_bar, W_vec))
            dφ_dx = float(np.dot(K_bar, φ_vector))

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)




           
            # exact fields
            exact_value = np.asarray(exact_funcs(x_local[0] +x_phys)).flatten()

            w_exact  = float(exact_value[0])
            th_exact = float(exact_value[1])
            M_exact  = float(exact_value[2])
            V_exact  = float(exact_value[3])

            # accumulate L2 errors
            L2_w  += w_g * J * (w_exact  - W)**2
            L2_th += w_g * J * (th_exact - φ)**2
            L2_M  += w_g * J * (M_exact  - M)**2
            L2_V  += w_g * J * (V_exact  - V)**2

            # accumulate exact norms
            L2_w_exact  += w_g * J * (w_exact )**2
            L2_th_exact += w_g * J * (th_exact)**2
            L2_M_exact  += w_g * J * (M_exact )**2
            L2_V_exact  += w_g * J * (V_exact )**2

    # Absolute L2 errors
    abs_L2 = (
        np.sqrt(L2_w),
        np.sqrt(L2_th),
        np.sqrt(L2_M),
        np.sqrt(L2_V),
    )

    # Relative errors
    rel_L2 = (
        np.sqrt(L2_w / L2_w_exact)   if L2_w_exact  > 0 else 0,
        np.sqrt(L2_th / L2_th_exact) if L2_th_exact > 0 else 0,
        np.sqrt(L2_M / L2_M_exact)   if L2_M_exact  > 0 else 0,
        np.sqrt(L2_V / L2_V_exact)   if L2_V_exact  > 0 else 0,
    )

    return abs_L2, rel_L2

# VEM Quadratic L2 norm computation
def compute_L2_and_relative_errors_VQ(U, Input_data):

    import numpy as np
    L,EI,GA, no_ele,no_guass_point,exact_funcs = Input_data
 
    le = L / no_ele

    

    # Gauss weights and points
    xi_points,Wp = np.polynomial.legendre.leggauss(no_guass_point)

    # Connectivity
    dof_conn = dof_quadratic(no_ele)
    node_coords = node_coords_linear(no_ele, L)
    N_func , N_bar_func = VEM_Shape_function_Quadratic()

    # Initialize L2 accumulators
    L2_w = 0.0
    L2_th = 0.0
    L2_M = 0.0
    L2_V = 0.0

    L2_w_exact = 0.0
    L2_th_exact = 0.0
    L2_M_exact = 0.0
    L2_V_exact = 0.0


    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2]
        dofs = dof_conn[e]
        u_e = U[dofs]                   # [W1, φ1, W2, φ2, W3, φ3]

                  

        
        W1, φ1, W2, φ2, W3, φ3 = u_e
        W_vec= np.array([W1,W3,W2])        # In formulation tthe order is W1(left node) , W3(right node) , W2(internal moment)
        φ_vector = np.array([φ1,φ3, φ2]) #In formulation tthe order is φ1(left node) , φ3(right node) , φ2(internal moment)

        # shape function derivatives
        J = le/2
       

        for gp in range(no_guass_point):
            ξ = xi_points[gp]
            w_g = Wp[gp]
            x_phys = le/2 * (1 + ξ)

            N = N_func(x_phys,le)
            N_bar = N_bar_func(x_phys,le)
            N     = np.array(N,     dtype=float).reshape(-1)
            N_bar =  np.array(N_bar, dtype=float).reshape(-1)

            W = float(np.dot(N, W_vec))
            φ = float(np.dot(N, φ_vector))

            dw_dx = float(np.dot(N_bar, W_vec))
            dφ_dx = float(np.dot(N_bar, φ_vector))

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)

        


            
            # exact fields
            exact_value = np.asarray(exact_funcs(x_local[0] +x_phys)).flatten()

            w_exact  = float(exact_value[0])
            th_exact = float(exact_value[1])
            M_exact  = float(exact_value[2])
            V_exact  = float(exact_value[3])

            # accumulate L2 errors
            L2_w  += w_g * J * (w_exact  - W)**2
            L2_th += w_g * J * (th_exact - φ)**2
            L2_M  += w_g * J * (M_exact  - M)**2
            L2_V  += w_g * J * (V_exact  - V)**2

            # accumulate exact norms
            L2_w_exact  += w_g * J * (w_exact )**2
            L2_th_exact += w_g * J * (th_exact)**2
            L2_M_exact  += w_g * J * (M_exact )**2
            L2_V_exact  += w_g * J * (V_exact )**2

    # Absolute L2 errors
    abs_L2 = (
        np.sqrt(L2_w),
        np.sqrt(L2_th),
        np.sqrt(L2_M),
        np.sqrt(L2_V),
    )

    # Relative errors
    rel_L2 = (
        np.sqrt(L2_w / L2_w_exact)   if L2_w_exact  > 0 else 0,
        np.sqrt(L2_th / L2_th_exact) if L2_th_exact > 0 else 0,
        np.sqrt(L2_M / L2_M_exact)   if L2_M_exact  > 0 else 0,
        np.sqrt(L2_V / L2_V_exact)   if L2_V_exact  > 0 else 0,
    )

    return abs_L2, rel_L2