"""
Timoshenko Beam FEM/VEM Package
================================

A package for finite element and virtual element analysis of Timoshenko beams.
"""

from .elements import (
    FEM_11, FEM_11_red, FEM_22,  
    FEM_21, 
    VEM_11, VEM_22,  VEM_21, VEM_Shape_function_Linear,VEM_Shape_function_Quadratic
)
from .assembly import Timosheko_beam, assemble_global_system, solve_system
from .utils import dof_linear, dof_quadratic, dof_mixed,node_coords_linear,node_coords_quadratic
from .plotting import compute_w_φ_M_V_FL,compute_w_φ_M_V_FM,compute_w_φ_M_V_FQ,compute_w_φ_M_V_VM,compute_w_φ_M_V_VQ