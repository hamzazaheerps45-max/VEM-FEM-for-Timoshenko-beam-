# Timoshenko Beam Analysis using FEM and VEM

This repository contains an implementation of the **Finite Element Method (FEM)** and **Virtual Element Method (VEM)** for the static analysis of **Timoshenko beam theory**.

The focus of this work is on:
- Verification of numerical formulations
- h-convergence studies
- Slenderness effects
- Comparison between FEM and VEM
- Symbolic derivation of the stiffness matrix

---


## Description of Examples

### 1. Cantilever Beam
- Static analysis of a Timoshenko cantilever beam
- Includes:
  - Comparison of results for different order and method
  - h-convergence study
  - Comparison of results for different slenderness ratios

### 2. Simply Supported Beam
- Static analysis of a simply supported Timoshenko beam
- Includes:
  - Comparison of results for different order and method
  - h-convergence study
  - Comparison of results for different slenderness ratios

### 3. Symbolic Stiffness Matrix
- Symbolic derivation of the Timoshenko beam stiffness matrix

---

## Numerical Methods

- **FEM**: Standard displacement-based formulation for Timoshenko beams
- **VEM**: Virtual Element formulation formulation for Timoshenko beams
- Comparison between FEM and VEM is performed through numerical experiments

---

## Key Features

- Modular and extensible source code
- Clear separation between solver and example problems
- Numerical convergence studies
- Slenderness parameter investigation
- Symbolic stiffness matrix derivation

---

## Requirements

- Programming language: * Python *
- Additional libraries:  *NumPy, SymPy*

---

## How to Run

1. Navigate to the `examples` directory
2. Run the desired example file:
   - Cantilever beam
   - Simply supported beam
   - Symbolic stiffness matrix
3. Modify parameters inside the example files to perform additional studies

---


## Author

*Muhammad Hamza*  
*University of Stuttgart*

---

## Reference

1. Wriggers, P. (2023). A locking-free virtual element formulation for Timoshenko beams. Computer Methods in Applied Mechanics and Engineering, Article 116234.
2. Wriggers, P. (2022). On a virtual element formulation for trusses and beams. Archive of Applied Mechanics, 92, 1655–1678.
3. Wriggers, P., Aldakheel, F., & Hudobivnik, B. (2024). Virtual Element Methods in Engineering Sciences Springer.
4. Öchsner, A. (2021). Classical Beam Theories of Structural Mechanics Springer.
5. von Scheven, M., Bischoff, M., & Ramm, E. (2024/2025). Computational Mechanics of Structures: Lecture Notes. Winter Term 2024/2025

---


## License

This project is intended for academic and research use.  
Add a license file if you plan to distribute or reuse the code.
