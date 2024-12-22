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
    # Combine the vectors from both subspaces into one matrix
    combined_matrix = np.hstack((vectors_l1, vectors_l2))
    
    # Use numpy's `linalg.matrix_rank` to determine the rank and identify linearly independent columns
    # We will use numpy's `svd` or `QR decomposition` (or Gaussian elimination method if required)
    U, S, Vt = np.linalg.svd(combined_matrix)
    
    # Find the rank, which tells how many linearly independent vectors we have
    rank = np.sum(S > 1e-10)  # Tolerance threshold for numerical precision issues

    print(f"Rank of the matrix (number of linearly independent vectors): {rank}")
    
    # The linearly independent vectors are the columns corresponding to the non-zero singular values
    # We find the indices of the non-zero singular values
    pivot_columns = np.where(S > 1e-10)[0]
    
    # The basis for the sum of subspaces corresponds to the pivot columns
    basis = combined_matrix[:, pivot_columns]
    
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
    
    print("\nEnter the vectors for subspace L1:")
    vectors_l1 = get_vectors(num_vectors_l1, dimension)
    if vectors_l1 is None:
        return
    
    print("\nEnter the vectors for subspace L2:")
    vectors_l2 = get_vectors(num_vectors_l2, dimension)
    if vectors_l2 is None:
        return
    
    # Compute the basis of
    # the sum of the subspaces
    basis = compute_basis_of_sum(vectors_l1, vectors_l2)
    
    # Print the basis vectors
    print_basis(basis)

if __name__ == "__main__":
    main()
