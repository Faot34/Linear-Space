import sympy as sp

def get_vectors(prompt, num_vectors, dimension):
    """
    Helper function to get vectors from the user.
    """
    vectors = []
    print(f"Enter {num_vectors} vectors for {prompt} (each vector has {dimension} entries):")
    for i in range(num_vectors):
        vector = list(map(int, input(f"Vector {i + 1}: ").strip().split()))
        if len(vector) != dimension:
            raise ValueError(f"Each vector must have exactly {dimension} entries!")
        vectors.append(vector)
    return vectors

def compute_intersection_basis(solution, subspace_2):
    """
    Compute the basis for the intersection using the solution vector and subspace 2.
    """
    # Extract the coefficients for Subspace 2 from the solution vector
    num_vectors_2 = len(subspace_2)
    coefficients_2 = solution[-num_vectors_2:]
    
    # Perform the linear combination of vectors in Subspace 2
    intersection_basis = sp.zeros(len(subspace_2[0]), 1)
    for coeff, vec in zip(coefficients_2, subspace_2):
        intersection_basis += coeff * sp.Matrix(vec)
    
    return intersection_basis

def main():
    # Step 1: Get dimensions and vectors
    dimension = int(input("Enter the dimension of the vectors: "))
    
    # Subspace 1
    num_vectors_1 = int(input("Enter the number of vectors in the first subspace: "))
    subspace_1 = get_vectors("first subspace", num_vectors_1, dimension)
    
    # Subspace 2
    num_vectors_2 = int(input("Enter the number of vectors in the second subspace: "))
    subspace_2 = get_vectors("second subspace", num_vectors_2, dimension)
    
    # Step 2: Construct the matrix
    # Place the vectors from Subspace 1 as columns
    matrix = sp.zeros(dimension, num_vectors_1 + num_vectors_2)
    for j, vec in enumerate(subspace_1):
        for i, value in enumerate(vec):
            matrix[i, j] = value
    
    # Place the additive inverses of the vectors from Subspace 2 as columns
    for j, vec in enumerate(subspace_2):
        for i, value in enumerate(vec):
            matrix[i, num_vectors_1 + j] = -value  # Additive inverse
    
    print("\nConstructed Matrix (with vectors as columns):")
    sp.pprint(matrix)
    
    # Step 3: Perform RREF
    rref_matrix, pivots = matrix.rref()
    
    print("\nMatrix in Reduced Row Echelon Form (RREF):")
    sp.pprint(rref_matrix)
    
    # Step 4: Parametrize the solution space
    num_cols = matrix.cols
    num_rows = matrix.rows
    free_vars = [i for i in range(num_cols) if i not in pivots]  # Non-pivot columns
    solutions = []
    
    print("\nParametrizing the solution space:")
    for free_var in free_vars:
        # Initialize a solution vector
        solution = [0] * num_cols
        solution[free_var] = 1  # Set the free variable to 1
        
        # Back-substitute using the RREF to solve for pivot variables
        for row, pivot_col in enumerate(pivots):
            solution[pivot_col] = -rref_matrix[row, free_var]
        
        solutions.append(solution)
        print(f"Solution vector for free variable {free_var}: {solution}")
    
    # Step 5: Compute the basis for the intersection
    print("\nComputing the basis for the intersection of the two subspaces:")
    intersection_basis_vectors = []
    for solution in solutions:
        # Compute the intersection basis using the solution and Subspace 2
        basis_vec = compute_intersection_basis(solution, subspace_2)
        intersection_basis_vectors.append(basis_vec)
    
    print("\nBasis for the Intersection of the Two Subspaces:")
    for i, basis_vec in enumerate(intersection_basis_vectors, start=1):
        print(f"Basis Vector {i}:")
        sp.pprint(basis_vec)

if __name__ == "__main__":
    main()
