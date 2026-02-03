"""
Utility functions for FEM/VEM analysis.
"""

import numpy as np




def dof_linear(n_elem):
    """Linear 2-node beam: [w1, θ1, w2, θ2]"""
    dof_conn = np.zeros((n_elem, 4), dtype=int)
    for e in range(n_elem):
        n1, n2 = e, e+1
        dof_conn[e] = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
    return dof_conn



def dof_quadratic(n_elem):
    """Quadratic 3-node beam: [w1,θ1, w2,θ2, w3,θ3]"""
    dof_conn = np.zeros((n_elem, 6), dtype=int)
    for e in range(n_elem):
        n1, n2, n3 = 2*e, 2*e+1, 2*e+2
        dof_conn[e] = [2*n1,2*n1+1, 2*n2,2*n2+1, 2*n3,2*n3+1]
    return dof_conn


def dof_mixed(n_elem):
    """
    Mixed element: quadratic w (3 nodes), linear φ (2 end nodes only)
    Local DOFs: [w1, φ1, w2, w3, φ3]
    
    Global DOF structure:
    - End nodes (even indices): have both w and φ
    - Middle nodes (odd indices): have only w
    """
    dof_conn = np.zeros((n_elem, 5), dtype=int)
    
    for e in range(n_elem):
        # Physical nodes for this element
        n1 = 2*e      # left end node
        n2 = 2*e + 1  # middle node
        n3 = 2*e + 2  # right end node
        
        # Compute global DOF indices
        # End nodes have 2 DOFs each (w, φ)
        # Middle nodes have 1 DOF (w only)
        
        # Count DOFs before node n1, n2, n3
        # Pattern: node 0→(w,φ), node 1→(w), node 2→(w,φ), node 3→(w), ...
        def global_dof(node):
            n_pairs = (node + 1) // 2  # number of complete (even,odd) pairs before this node
            if node % 2 == 0:  # even node (has w and φ)
                w_dof = n_pairs * 3
                phi_dof = w_dof + 1
                return w_dof, phi_dof
            else:  # odd node (has w only)
                return n_pairs * 3 - 1, None
        
        w1_dof, phi1_dof = global_dof(n1)
        w2_dof, _ = global_dof(n2)
        w3_dof, phi3_dof = global_dof(n3)
        
        dof_conn[e] = [w1_dof, phi1_dof, w2_dof, w3_dof, phi3_dof]
    
    return dof_conn



def node_coords_linear(n_elem, L):
    x_coords = np.linspace(0, L, n_elem+1)  # global node coordinates
    node_coords = np.zeros((n_elem, 2), dtype=float)

    for e in range(n_elem):
        node_coords[e] = x_coords[[e, e+1]]

    return node_coords



def node_coords_quadratic(n_elem, L):
    x_coords = np.linspace(0, L, 2*n_elem+1)  # global nodes
    node_coords = np.zeros((n_elem, 3), dtype=float)

    for e in range(n_elem):
        node_coords[e] = x_coords[[2*e, 2*e+1, 2*e+2]]

    return node_coords
