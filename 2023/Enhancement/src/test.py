import numpy as np
from math import log2
from skimage.feature import greycomatrix, greycoprops

def calculate_second_order_entropy(co_occurrence_matrix):
    # Normalize the co-occurrence matrix
    normalized_matrix = co_occurrence_matrix / np.sum(co_occurrence_matrix)

    # Calculate marginal probabilities
    row_sums = np.sum(normalized_matrix, axis=1)
    col_sums = np.sum(normalized_matrix, axis=0)

    # Calculate the second-order entropy
    second_order_entropy = 0.0
    for i in range(len(co_occurrence_matrix)):
        for j in range(len(co_occurrence_matrix)):
            if normalized_matrix[i, j] > 0.0:
                joint_prob = normalized_matrix[i, j]
                marginal_prob_i = row_sums[i]
                marginal_prob_j = col_sums[j]
                second_order_entropy += joint_prob * log2(joint_prob / (marginal_prob_i * marginal_prob_j))
                print("===========")
                print(marginal_prob_i)
                print(marginal_prob_j)
                print(joint_prob)
                print(log2(joint_prob / (marginal_prob_i * marginal_prob_j)))

    return -second_order_entropy

# Example co-occurrence matrix (3x3)
co_occurrence_matrix = np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]])

second_order_entropy = greycoprops(co_occurrence_matrix, 'entropy')[0, 0]
print("Second-order entropy:", second_order_entropy)