import numpy as np




from .utils import dof_linear, dof_quadratic, dof_mixed ,node_coords_linear ,node_coords_quadratic
from.elements import VEM_Shape_function_Linear,VEM_Shape_function_Quadratic

def compute_w_φ_M_V_FL(U, Input_data):
    

    L,EI,GA, no_ele,no_points = Input_data
    le = L/no_ele

    

    xi_points  = np.linspace(-1.0, 1.0, no_points)

    dof_conn = dof_linear(no_ele)
    node_coords = node_coords_linear(no_ele, L)

    element_results = []

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2]
        dofs = dof_conn[e]
        u_e = U[dofs]                  # [W1, φ1, W2, φ2]

        W1, φ1, W2, φ2 = u_e

        # shape function derivatives
        dN1_dξ = -0.5
        dN2_dξ =  0.5
        J_inv = 2 / le

        dN1_dx = dN1_dξ * J_inv
        dN2_dx = dN2_dξ * J_inv

        # shape functions
        def N1(ξ): return 0.5*(1-ξ)
        def N2(ξ): return 0.5*(1+ξ)

        x_vals = []
        w_vals = []
        φ_vals = []
        M_vals = []
        V_vals = []

        for ξ in xi_points:

            N1ξ = N1(ξ)
            N2ξ = N2(ξ)

            W = N1ξ*W1 + N2ξ*W2
            φ = N1ξ*φ1 + N2ξ*φ2

            dw_dx = dN1_dx*W1 + dN2_dx*W2
            dφ_dx = dN1_dx*φ1 + dN2_dx*φ2

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)

            x_phys = N1ξ*x_local[0] + N2ξ*x_local[1]

            x_vals.append(x_phys)
            w_vals.append(W)
            φ_vals.append(φ)
            M_vals.append(M)
            V_vals.append(V)

        element_results.append((x_vals, w_vals, φ_vals, M_vals, V_vals))

    return element_results


def compute_w_φ_M_V_FQ(U, Input_data):

    L,EI,GA, no_ele,no_points = Input_data
    le = L/no_ele

    xi_points  = np.linspace(-1.0, 1.0, no_points)

    dof_conn = dof_quadratic(no_ele)
    node_coords = node_coords_quadratic(no_ele, L)

    element_results = []

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2,x3]
        dofs = dof_conn[e]
        u_e = U[dofs]                  

        W1, φ1, W2, φ2, W3, φ3 = u_e

        # shape function derivatives
       
        J_inv = 2 / le

      

        # shape functions
        def N1(ξ): return ξ*(ξ-1)/2
        def N2(ξ): return (1-ξ**2)
        def N3(ξ): return ξ*(ξ+1)/2
        def dN1dx(ξ): return J_inv*(2*ξ-1)/2
        def dN2dx(ξ): return J_inv*(-2*ξ)
        def dN3dx(ξ): return J_inv*(2*ξ+1)/2

        x_vals = []
        w_vals = []
        φ_vals = []
        M_vals = []
        V_vals = []

        for ξ in xi_points:

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

            x_vals.append(x_phys)
            w_vals.append(W)
            φ_vals.append(φ)
            M_vals.append(M)
            V_vals.append(V)

        element_results.append((x_vals, w_vals, φ_vals, M_vals, V_vals))

    return element_results


def compute_w_φ_M_V_FM(U,Input_data):

    L,EI,GA, no_ele,no_points = Input_data
    le = L/no_ele

    xi_points  = np.linspace(-1.0, 1.0, no_points)

    dof_conn = dof_mixed(no_ele)
    node_coords = node_coords_quadratic(no_ele, L)

    element_results = []

    for e in range(no_ele):

        x_local = node_coords[e]        # [x1, x2,x3]
        dofs = dof_conn[e]
        u_e = U[dofs]                  

        W1, φ1, W2, W3, φ3 = u_e

        # shape function derivatives
       
        J_inv = 2 / le

      

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


        x_vals = []
        w_vals = []
        φ_vals = []
        M_vals = []
        V_vals = []

        for ξ in xi_points:

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

            x_vals.append(x_phys)
            w_vals.append(W)
            φ_vals.append(φ)
            M_vals.append(M)
            V_vals.append(V)

        element_results.append((x_vals, w_vals, φ_vals, M_vals, V_vals))

    return element_results








def compute_w_φ_M_V_VM(U, Input_data):
    L,EI,GA, no_ele,no_points = Input_data
    le = L/no_ele




    dof_conn = dof_mixed(no_ele)
    node_coords = node_coords_linear(no_ele, L)
    K_func , K_bar_func = VEM_Shape_function_Linear()
    N_func , N_bar_func = VEM_Shape_function_Quadratic()

    element_results = []

    for e in range(no_ele):

        x_local = node_coords[e]
        x_points = np.linspace(0.0, le, no_points)
        dofs = dof_conn[e]
        u_e = U[dofs]                  

        W1, φ1, W2, W3, φ3 = u_e
        W_vec= np.array([W1,W3,W2])
        φ_vector = np.array([φ1,φ3])



        x_vals = []
        w_vals = []
        φ_vals = []
        M_vals = []
        V_vals = []

        for x_phys in x_points:
            N = N_func(x_phys,le)
            N_bar = N_bar_func(x_phys,le)
            K = K_func(x_phys,le)
            K_bar= K_bar_func(x_phys,le)
            
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

            

            x_vals.append(x_local[0]+x_phys)
            w_vals.append(W)
            φ_vals.append(φ)
            M_vals.append(M)
            V_vals.append(V)

        element_results.append((x_vals, w_vals, φ_vals, M_vals, V_vals))

    return element_results





def compute_w_φ_M_V_VQ(U, Input_data):

    L,EI,GA, no_ele,no_points = Input_data
    le = L/no_ele




    dof_conn = dof_quadratic(no_ele)
    node_coords = node_coords_linear(no_ele, L)
    N_func , N_bar_func = VEM_Shape_function_Quadratic()

    element_results = []

    for e in range(no_ele):

        x_local = node_coords[e]
        x_points = np.linspace(0.0, le, no_points)
    
        dofs = dof_conn[e]
        u_e = U[dofs]                  

        W1, φ1, W2, φ2, W3, φ3 = u_e
        W_vec= np.array([W1,W3,W2])
        φ_vector = np.array([φ1,φ3, φ2])



        x_vals = []
        w_vals = []
        φ_vals = []
        M_vals = []
        V_vals = []

        for x_phys in x_points:
            N = N_func(x_phys,le)
            N_bar = N_bar_func(x_phys,le)
            N     = np.array(N,     dtype=float).reshape(-1)
            N_bar = np.array(N_bar, dtype=float).reshape(-1)

            W = float(np.dot(N, W_vec))
            φ = float(np.dot(N, φ_vector))

            dw_dx = float(np.dot(N_bar, W_vec))
            dφ_dx = float(np.dot(N_bar, φ_vector))
                        
        

            M = EI * dφ_dx
            V = GA * (dw_dx - φ)

            

            x_vals.append(x_local[0]+x_phys)
            w_vals.append(W)
            φ_vals.append(φ)
            M_vals.append(M)
            V_vals.append(V)

        element_results.append((x_vals, w_vals, φ_vals, M_vals, V_vals))

    return element_results