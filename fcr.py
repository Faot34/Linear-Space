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

    # Step 5: Perform linear combination to find the fundamental solution space
    print("\nCalculating the linear combinations for the solution space:")

    basis_vectors = []
    for sol in solutions:
        vector = sp.Matrix(num_unknowns, 1, sol)  # Convert solution vector to a column vector
        basis_vectors.append(vector)

    for i, vec in enumerate(basis_vectors):
        print(f"Basis vector {i + 1}: ")
        sp.pprint(vec)

    # Combine the basis vectors into a matrix form
    fsr_matrix = sp.Matrix.hstack(*basis_vectors)

    print("\nFundamental Solution System (ФСР) Matrix (basis vectors as columns):")
    sp.pprint(fsr_matrix)

if __name__ == "__main__":
    main()
