'''
Hoshen-Kopelman Algorithm for Cluster Labeling

The Hoshen-Kopelman algorithm is a method for identifying and labeling clusters in a two-dimensional matrix. 
Originally developed for percolation theory in statistical physics. This code is a prelude to applications in the 2d XY model.
The algorithm operates on a binary matrix where each element is either occupied (1) or unoccupied (0). 
It systematically labels clusters of connected occupied elements, assigning each cluster a unique identifier. 
The core idea is to traverse the matrix and label clusters based on their connectivity.

Initialization:
A global variable, often called the "running variable," is used to assign unique labels to clusters.

Traversal:
The matrix is traversed row by row, left to right.
For each occupied element encountered:
If it is isolated (not connected to any occupied element above or to its left), a new cluster is created, and the running variable is incremented.
If it is connected to an occupied element above or to its left, it is assigned the label of the connected element.

Operations:
When two clusters need to be merged, the root label of one cluster is set as the parent of the root label of the other cluster.

Label Renaming:
After traversal, the labels are renamed to ensure that each cluster has a unique identifier.
The find_original function is used to find the root label of a cluster.

Cluster Counting:
The algorithm counts the number of distinct clusters and records their sizes in the cluster_sizes dictionary.
Implementation:

Usage:
To use the algorithm, provide a binary matrix where 1 represents an occupied element and 0 represents an unoccupied element. 
The code will then label and count the clusters within the matrix.
'''
from itertools import product

# Helper function to print a dictionary in a tabular format.
def print_dictionary(data_dict):
    """
    Prints a dictionary in a tabular format.
    :param data_dict: The dictionary to be printed.
    """
    for key, value in data_dict.items():
        print("{0}\t|\t{1}".format(key, value))

# Helper function to print a 2D matrix without tabulate.
def print_2d_matrix(matrix):
    """
    Prints a 2D matrix without using tabulate.
    :param matrix: The 2D matrix to be printed.
    """
    for row in matrix:
        print("\t".join(map(str, row)))

# Dictionary to store cluster sizes.
cluster_sizes = {}

# Counter to track the running variable for cluster labels.
running_variable = 0

def create_cluster():
    """
    Creates a new cluster and returns its label.
    """
    global running_variable

    running_variable += 1
    cluster_sizes[running_variable] = 1
    return running_variable


def add_to_cluster(label):
    """
    Adds an element to the specified cluster and returns the label.
    """
    cluster_sizes[find_original(label)] += 1
    return label


def find_original(label):
    """
    Finds the original cluster label to which the given label belongs.
    """
    while cluster_sizes[label] < 0:
        label = -cluster_sizes[label]
    
    return label


def unify_clusters(above, left):
    """
    Unifies two clusters and returns the label of the unified cluster.
    """
    original_cluster_above = find_original(above)

    if above == left:
        return add_to_cluster(above)

    if find_original(left) == original_cluster_above:
        cluster_sizes[original_cluster_above] += 1
        return above

    cluster_sizes[original_cluster_above] += cluster_sizes[find_original(left)] + 1

    if cluster_sizes.get(left, False):
        previous_root = cluster_sizes[left]
        if previous_root < 0:
            cluster_sizes[-previous_root] = -original_cluster_above
    
    cluster_sizes[left] = -original_cluster_above
    return above


def hoshen_kopelman(matrix):
    """
    Applies the Hoshen-Kopelman algorithm to label clusters in the matrix.
    """
    global running_variable

    for i, j in product(range(len(matrix)), range(len(matrix[0]))):
        if matrix[i][j] == 0:
            continue

        occupied_above = False if i == 0 else (matrix[i-1][j] != 0)
        occupied_left = False if j == 0 else (matrix[i][j-1] != 0)

        if not occupied_above and not occupied_left:
            running_variable = create_cluster()
            matrix[i][j] = running_variable

        elif occupied_above != occupied_left:
            matrix[i][j] = add_to_cluster(matrix[i-1][j] if occupied_above else matrix[i][j-1])
        
        elif occupied_above and occupied_left:
            above = matrix[i-1][j]
            left = matrix[i][j-1]
            matrix[i][j] = unify_clusters(above, left)

def rename_clusters(matrix):
    """
    Replaces the values in the matrix with the root labels of the clusters.
    """
    for i, j in product(range(len(matrix)), range(len(matrix[0]))):
        if matrix[i][j] != 0:  
            matrix[i][j] = find_original(matrix[i][j])

def count_distinct_clusters():
    """
    Counts the distinct clusters in the matrix.
    """
    cluster_count = sum(1 for key in cluster_sizes.keys() if cluster_sizes[key] > 0)
    return cluster_count

def count_clusters(input_matrix):
    """
    Counts clusters in the given matrix and prints the results.
    """
    global cluster_sizes
    global running_variable

    matrix = [[1 if item else 0 for item in row] for row in input_matrix]
    cluster_sizes.clear()
    running_variable = 0

    print("The matrix you have entered is:")
    print_2d_matrix(matrix)

    print("\nStarting labeling... ")
    hoshen_kopelman(matrix)

    print("\nRenaming clusters... ")
    rename_clusters(matrix)

    cluster_count = count_distinct_clusters()

    print("\nLabeling complete. The matrix is now:")
    print_2d_matrix(matrix)
    
    print("\nEncountered {0} clusters of which, {1} were distinct. Their sizes are as follows:".format(max(map(max, matrix)), cluster_count))
    print_dictionary(cluster_sizes)
    print("\n")

def main():
    # Example usage of the Hoshen-Kopelman algorithm and printing functions
    # (You can replace this with your actual matrix and desired operations)
    matrix = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]

    count_clusters(matrix)

if __name__ == "__main__":
    main()
