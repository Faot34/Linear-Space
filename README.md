# Matrix Operations Toolkit Documentation

This document provides detailed information on how to use the Python scripts for various matrix and vector operations, including row reduction, solving systems of linear equations, and computing the bases of subspaces. These scripts rely on libraries like SymPy and NumPy, so please ensure your Python environment is set up correctly before running them.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Scripts Overview](#scripts-overview)
   - [Row Reduction](#row-reduction)
   - [System of Linear Equations Solver](#system-of-linear-equations-solver)
   - [Basis of Subspaces](#basis-of-subspaces)
   - [Intersection of Subspaces](#intersection-of-subspaces)
4. [Common Issues](#common-issues)

---

## Requirements

- Python 3.6+
- SymPy
- NumPy

Ensure that your Python installation includes `pip`, the package manager, to install the required libraries.

---

## Installation

To set up your environment, follow these steps:

1. Install Python from the official Python website: [https://www.python.org/](https://www.python.org/)

2. Ensure `pip` is installed and added to your system's PATH. Run the following command in your terminal or command prompt to check:

   ```bash
   pip --version
   ```

   If `pip` is not installed, refer to the [pip installation guide](https://pip.pypa.io/en/stable/installation/).

3. Install the required libraries using pip:

   ```bash
   pip install sympy numpy
   ```

---

## Scripts Overview

### Row Reduction

#### Description:

This script performs manual row reduction on a matrix and converts it into Reduced Row Echelon Form (RREF). You can track each operation (row swaps, normalization, and elimination) performed on the matrix.

#### Code:

```python
import sympy as sp

def row_reduction():
    # Step 1: Ask the user for the matrix dimensions
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    # Step 2: Input the matrix coefficients
    print("Enter the coefficients of the matrix row by row, separated by spaces.")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i + 1}: ").strip().split()))
        if len(row) != cols:
            raise ValueError(f"Each row must have exactly {cols} entries!")
        matrix.append(row)

    # Convert the input into a SymPy matrix
    A = sp.Matrix(matrix)

    print("\nInitial Matrix:")
    sp.pprint(A)

    # Step 3: Perform row reduction manually
    def print_operation(op_desc, mat):
        print(f"\n{op_desc}:")
        sp.pprint(mat)

    for pivot_row in range(min(rows, cols)):
        # Step 3.1: Ensure pivot is non-zero; swap rows if necessary
        if A[pivot_row, pivot_row] == 0:
            for swap_row in range(pivot_row + 1, rows):
                if A[swap_row, pivot_row] != 0:
                    A.row_swap(pivot_row, swap_row)
                    print_operation(f"Swapped row {pivot_row + 1} with row {swap_row + 1}", A)
                    break

        # If pivot remains zero, skip this column
        if A[pivot_row, pivot_row] == 0:
            continue

        # Step 3.2: Normalize the pivot row
        pivot = A[pivot_row, pivot_row]
        A.row_op(pivot_row, lambda v, _: v / pivot)
        print_operation(f"Normalized row {pivot_row + 1}", A)

        # Step 3.3: Eliminate entries below the pivot
        for target_row in range(pivot_row + 1, rows):
            multiplier = A[target_row, pivot_row]
            A.row_op(target_row, lambda v, j: v - multiplier * A[pivot_row, j])
            print_operation(f"Eliminated entry in row {target_row + 1}, column {pivot_row + 1}", A)

    # Step 4: Back substitution to eliminate above pivots (if needed)
    for pivot_row in range(min(rows, cols) - 1, -1, -1):
        if A[pivot_row, pivot_row] == 0:
            continue

        for target_row in range(pivot_row - 1, -1, -1):
            multiplier = A[target_row, pivot_row]
            A.row_op(target_row, lambda v, j: v - multiplier * A[pivot_row, j])
            print_operation(f"Eliminated entry in row {target_row + 1}, column {pivot_row + 1}", A)

    print("\nFinal Reduced Row Echelon Form (RREF):")
    sp.pprint(A)

if __name__ == "__main__":
    row_reduction()
```

---

### System of Linear Equations Solver

#### Description:

This script solves systems of linear equations using SymPy. It computes the Reduced Row Echelon Form (RREF) of the augmented matrix and parametrizes the solution space.

#### Code:

```python
import sympy as sp

def main():
    # Step 1: Ask the user for the number of unknowns and equations
    num_unknowns = int(input("Enter the number of unknowns: "))
    num_equations = int(input("Enter the number of equations: "))

    if num_equations > num_unknowns:
        print("Warning: The system might be inconsistent (more equations than unknowns).")

    # Step 2: Input the coefficients of the equations
    print("Enter the coefficients of each equation, separated by spaces.")
    matrix = []
    for i in range(num_equations):
        row = list(map(float, input(f"Equation {i + 1}: ").strip().split()))
        if len(row) != num_unknowns:
            raise ValueError(f"Each equation must have exactly {num_unknowns} coefficients!")
        matrix.append(row)

    # Convert the input into a SymPy matrix
    A = sp.Matrix(matrix)

    # Step 3: Perform Row Reduction to Reduced Row Echelon Form (RREF)
    rref_matrix, pivots = A.rref()

    print("\nMatrix in Reduced Row Echelon Form (RREF):")
    sp.pprint(rref_matrix)

    # Step 4: Parametrize the solution space
    print("\nParametrizing the solution space:")

    # Identify free and leading variables
    free_vars = [i for i in range(num_unknowns) if i not in pivots]

    solutions = []  # This will store the solutions representing the fundamental system of solutions

    for free_var in free_vars:
        # Initialize a solution vector (all zeroes initially)
        solution = [0] * num_unknowns

        # Set the free variable to 1
        solution[free_var] = 1

        # Back-substitute to solve for pivot variables in terms of the free variable
        for row, pivot_col in enumerate(pivots):
            solution[pivot_col] = -rref_matrix[row, free_var]

        solutions.append(solution)
        print(f"Solution vector for free variable {free_var}: {solution}")

    # Combine the basis vectors into a matrix form
    basis_vectors = [sp.Matrix(num_unknowns, 1, sol) for sol in solutions]

    print("\nBasis for the solution space:")
    for vec in basis_vectors:
        sp.pprint(vec)

if __name__ == "__main__":
    main()
```

---

### Basis of Subspaces

#### Description:

This script computes the basis of the sum of two subspaces. It identifies the linearly independent vectors from the combined subspaces using NumPy.

#### Code:

```python
import numpy as np

def get_vectors(num_vectors, dimension):
    vectors = []
    for i in range(num_vectors):
        print(f"Enter the components of vector {i+1} (space-separated):")
        vector = list(map(float, input().split()))
        if len(vector) != dimension:
            print(f"Error: Vector must have {dimension} components.")
            return None
        vectors.append(vector)
    return np.array(vectors).T  # Transpose to get vectors as columns

def compute_basis_of_sum(vectors_l1, vectors_l2):
    combined_matrix = np.hstack((vectors_l1, vectors_l2))
    U, S, Vt = np.linalg.svd(combined_matrix)
    rank = np.sum(S > 1e-10)
    basis = combined_matrix[:, :rank]
    return basis

def print_basis(basis):
    print("The basis for the sum of the subspaces is:")
    for i, vector in enumerate(basis.T):
        print(f"v{i+1} = {vector}")

def main():
    print("Enter the dimension of the vectors:")
    dimension = int(input())
    print("Enter the number of vectors in subspace L1:")
    num_vectors_l1 = int(input())
    print("Enter the number of vectors in subspace L2:")
    num_vectors_l2 = int(input())
    vectors_l1 = get_vectors(num_vectors_l1, dimension)
    vectors_l2 = get_vectors(num_vectors_l2, dimension)
    basis = compute_basis_of_sum(vectors_l1, vectors_l2)
    print_basis(basis)

if __name__ == "__main__":
    main()
```

