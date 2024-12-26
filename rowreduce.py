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
