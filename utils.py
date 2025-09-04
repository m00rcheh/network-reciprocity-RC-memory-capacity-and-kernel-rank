import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from scipy.sparse import rand
# import random
# import math
# import seaborn as sb
# from scipy.io import savemat
from numba import jit
# import math
# from scipy.linalg import schur
from scipy.sparse.csgraph import laplacian
import os
import community as community_louvain
import bct
from scipy.linalg import norm, svd
from scipy.optimize import curve_fit
from scipy import linalg
from scipy.sparse.csgraph import connected_components
from numpy import linalg as LA


def convert_graph_to_adjacency_matrix(g):
    return nx.adjacency_matrix(g).toarray()


def convert_adjacency_matrix_to_graph(w):
    return nx.from_numpy_array(w)



def adjust_reciprocity_binary(adj_matrix, desired_reciprocity):
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = adj_matrix.astype('bool')
    L = adj_matrix.sum()
    current_r = compute_reciprocity_binary(adj_matrix)
    if np.isclose(current_r, desired_reciprocity, atol=1e-5):
        pass
    else:
        symmetric_indices = np.column_stack(np.where((adj_matrix == 1) * (adj_matrix.T == 1) *
                                                     np.triu(np.ones_like(adj_matrix), k=1)))
        num_symmetric_indices = symmetric_indices.shape[0]  # the network has num_symmetric_indices*2 reciprocal link
        num_reciprocal = int(desired_reciprocity * L)  # desired number of reciprocal links

        diff = 2 * num_symmetric_indices - num_reciprocal
        symmetric_zeros = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 0) *
                                                   np.triu(np.ones_like(adj_matrix), k=1)))

        if diff >= 0:
            if symmetric_zeros.shape[0] < np.abs(diff):
                print('Error: density is too high. Sparsify your network and try again.')
            else:
                # remove reciprocal links
                num_to_remove = num_symmetric_indices - int(num_reciprocal/2)# - 1
                selected_indices = np.random.choice(num_symmetric_indices, num_to_remove, replace=False)
                selected_ones_indices = symmetric_indices[selected_indices[:num_to_remove]]
                for index in selected_ones_indices:
                    i, j = index
                    if np.random.rand(1, 1) >= 0.5:
                        adj_matrix[i, j] = 0
                        adj_matrix[j, i] = 1
                    else:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 0
                symmetric_zeros = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 0) *
                                                           np.triu(np.ones_like(adj_matrix), k=1)))
                if symmetric_zeros.shape[0] >= int(num_to_remove):
                    selected_zero_indices_extra = np.random.choice(symmetric_zeros.shape[0],
                                                                   int(num_to_remove),
                                                                   replace=False)
                    selected_zero_to_ones_indices = symmetric_zeros[selected_zero_indices_extra]
                    for index in selected_zero_to_ones_indices:
                        i, j = index
                        if np.random.rand(1, 1) >= 0.5:
                            adj_matrix[i, j] = 0
                            adj_matrix[j, i] = 1
                        else:
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 0
        else:
            num_single_to_reciprocal = np.abs(diff/2) + 1
            Rest = np.abs(adj_matrix.astype('bool') ^ (adj_matrix.astype('bool')).T)
            num_single = 0.5 * np.abs(Rest).sum()
            if num_single >= 2 * num_single_to_reciprocal:
                single_links_indices = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 1)))
                selected_single_link_indices = np.random.choice(int(num_single), int(2 * num_single_to_reciprocal),
                                                                replace=False)
                selected_links_indices = (
                    single_links_indices)[selected_single_link_indices[:int(num_single_to_reciprocal)]]
                for index in selected_links_indices:
                    i, j = index
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                selected_links_indices_to_remove = single_links_indices[
                    selected_single_link_indices[int(num_single_to_reciprocal):]]
                for index in selected_links_indices_to_remove:
                    i, j = index
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
            else:
                single_links_indices = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 1)))
                for index in single_links_indices:
                    i, j = index
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                num_reciprocal_to_remove = int(single_links_indices.shape[0] / 2)
                symmetric_indices = np.column_stack(np.where((adj_matrix == 1) * (adj_matrix.T == 1) *
                                                            np.triu(np.ones_like(adj_matrix), k=1)))
                selected_indices = np.random.choice(symmetric_indices.shape[0], num_reciprocal_to_remove, replace=False)
                selected_ones_indices = symmetric_indices[selected_indices[:num_reciprocal_to_remove]]
                for index in selected_ones_indices:
                    i, j = index
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
                # print('Error: not enough single links to reciprocate. Run advanced algorithm')
    return adj_matrix


@jit(nopython=True)
def calculate_W1_and_r(W, L):
    np.fill_diagonal(W, 0)
    W1 = np.minimum(W, W.T)
    r = W1.sum() / L
    return W1, r


def compute_reciprocity_weighted(adj_matrix):
    np.fill_diagonal(adj_matrix, 0)
    L = adj_matrix.sum()
    W1 = np.minimum(adj_matrix, adj_matrix.T)
    # W2 = adj_matrix - W1
    return np.round(W1.sum() / L, 2)


@jit(nopython=True)
def adjust_matrix_gradually(W, L, target_r_new, increment=0.01, max_iter=1000, max_value=10, min_value=0.01):
    n = W.shape[0]
    current_W = W.copy()

    for _ in range(max_iter):
        # Compute current W1 and r
        current_W1, current_r = calculate_W1_and_r(current_W, L)

        if np.isclose(current_r, target_r_new, atol=1e-5) and np.abs(current_W.sum() - L) / L < 1e-5:
            break

        # Compute scaling factor for the current step
        if current_r == 0:
            scaling_factor = 0
        else:
            scaling_factor = target_r_new / current_r

        # Gradual scaling
        if scaling_factor > 1:
            scaling_factor = min(scaling_factor, 1 + increment)
        elif scaling_factor < 1:
            scaling_factor = max(scaling_factor, 1 - increment)

        # Apply scaling factor to elements contributing to W1
        W1_contributing_indices = np.argwhere(current_W <= current_W.T)

        for i, j in W1_contributing_indices:
            if i != j:
                current_W[i, j] = min(max_value, current_W[i, j] * scaling_factor)

        # Update W1 after scaling
        current_W1, current_r = calculate_W1_and_r(current_W, L)

        # Adjust non-contributing elements to maintain the total sum L
        total_increment = current_W.sum() - L
        if np.abs(total_increment) > 1e-5:
            non_contributing_indices = np.argwhere(current_W > current_W.T)
            np.random.shuffle(non_contributing_indices)
            total_decrement = 0
            for i, j in non_contributing_indices:
                if total_decrement >= total_increment:
                    break
                if current_W[i, j] > min_value:
                    decrease = min(current_W[i, j] - min_value,
                                   (total_increment - total_decrement) / len(non_contributing_indices))
                    current_W[i, j] -= decrease
                    total_decrement += decrease

    # Ensure final adjustment
    # if np.abs(current_W.sum() - L) > 1e-5:
    #     raise ValueError("Sum of W has changed!")
    return current_W


def adjust_matrix(W, L, target_r_new):
    current_W1, current_r = calculate_W1_and_r(W, L)
    scaling_factor = target_r_new / (current_r + 0.0001)
    v = np.float64(scaling_factor * current_W1)
    residual = current_W1.sum() - v.sum()
    current_W2 = W - current_W1
    positive_indices = current_W2 > 0
    count = np.sum(positive_indices)
    current_W2[positive_indices] += residual / count
    current_W = v + current_W2

    return current_W


def adjust_reciprocity_weighted(adj_matrix, desired_reciprocity, num_iter=2000):
    np.fill_diagonal(adj_matrix, 0)
    L = adj_matrix.sum()
    W1, r = calculate_W1_and_r(adj_matrix, L)
    if desired_reciprocity <= r:
        if np.isclose(r, 1.0, atol=1e-5) and np.isclose(desired_reciprocity, 1.0, atol=1e-5):
            # Both r and desired_reciprocity are approximately 1.0
            pass
        elif np.isclose(r, 1.0, atol=1e-5):
            non_zero_elements = adj_matrix[adj_matrix != 0]
            # Check if there are non-zero elements to avoid errors
            if non_zero_elements.size == 0:
                raise ValueError("Matrix contains no non-zero elements.")
            min_non_zero_value = np.min(non_zero_elements)
            perturbation = np.random.rand(*adj_matrix.shape) * min_non_zero_value #0.0001
            non_zero_mask = adj_matrix != 0
            perturbation_values = perturbation[non_zero_mask]
            mean_perturbation = np.mean(perturbation_values)
            # Adjust perturbation to have zero mean
            adjusted_perturbation = perturbation - mean_perturbation
            # Apply adjusted perturbation only to non-zero entries in adj_matrix
            adj_matrix[non_zero_mask] += adjusted_perturbation[non_zero_mask]
            # perturbation -= np.mean(perturbation)  # Adjust to have zero mean

            # adj_matrix[non_zero_mask] += perturbation[non_zero_mask]
            # adj_matrix = adj_matrix + perturbation
            adj_matrix = adjust_matrix(adj_matrix, L, desired_reciprocity)
        else:
            adj_matrix = adjust_matrix(adj_matrix, L, desired_reciprocity)
    else:
        adj_matrix = adjust_matrix_gradually(adj_matrix, L, desired_reciprocity, increment=0.01,
                                                 max_iter=num_iter, max_value=1000, min_value=0.01)
        hat_W1, hat_r_new = calculate_W1_and_r(adj_matrix, L)

    return adj_matrix


def convert_binary_to_weighted_matrix(X):
    return X * np.random.uniform(0.1, 1, X.shape)


def compute_density(adjmatrix):
    adjmatrix = adjmatrix.astype('bool')
    return adjmatrix.sum() / (adjmatrix.shape[0] * (adjmatrix.shape[0] - 1))

def compute_reciprocity_binary(adjmatrix):
    adjmatrix = adjmatrix.astype('bool')
    L = adjmatrix.sum()
    if L == 0:
        reciprocity = 0
    else:
        Rest = np.abs(adjmatrix ^ adjmatrix.T)
        Lsingle = 0.5*Rest.sum()
        reciprocity = np.float64(L-Lsingle) / L

    return reciprocity


def find_spectral_radius(adjacency_matrix):
    eigenvalues = np.linalg.eigvals(adjacency_matrix)
    spectral_radius = np.max(np.abs(eigenvalues))
    return spectral_radius


def find_spectral_norm(adjacency_matrix):
    # Compute singular values
    singular_values = np.linalg.svd(adjacency_matrix, compute_uv=False)
    # Spectral norm is the largest singular value
    spectral_norm = max(singular_values)
    return spectral_norm

def find_matrix_rank(adjacency_matrix):
    rank = np.linalg.matrix_rank(adjacency_matrix.astype(float))
    return rank


def find_singular_value_decay(adj_matrix, plot=True):
    
    # Compute SVD
    U, s, Vt = svd(adj_matrix)

    # Create x values (indices)
    x = np.arange(1, len(s) + 1)

    # Normalize singular values
    s_norm = s / s[0]

    # Define fitting functions
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    def power_law(x, a, b):
        return a * x ** (-b)

    # Fit both exponential and power law
    try:
        popt_exp, _ = curve_fit(exp_decay, x, s_norm)
        exp_fit = exp_decay(x, *popt_exp)

        # Fit power law to log-log data
        popt_power, _ = curve_fit(power_law, x, s_norm)
        power_fit = power_law(x, *popt_power)

        # Calculate R-squared values
        residuals_exp = s_norm - exp_fit
        ss_res_exp = np.sum(residuals_exp ** 2)
        ss_tot = np.sum((s_norm - np.mean(s_norm)) ** 2)
        r2_exp = 1 - (ss_res_exp / ss_tot)

        residuals_power = s_norm - power_fit
        ss_res_power = np.sum(residuals_power ** 2)
        r2_power = 1 - (ss_res_power / ss_tot)

    except RuntimeError:
        print("Warning: Curve fitting failed")
        popt_exp = [np.nan, np.nan]
        popt_power = [np.nan, np.nan]
        r2_exp = np.nan
        r2_power = np.nan

    if plot:
        plt.figure(figsize=(12, 6))

        # Plot original data
        plt.subplot(121)
        plt.plot(x, s_norm, 'ko-', label='Singular values')
        plt.plot(x, exp_fit, 'r-', label=f'Exp fit (rate={popt_exp[1]:.3f})')
        plt.plot(x, power_fit, 'b-', label=f'Power fit (exp={popt_power[1]:.3f})')
        plt.xlabel('Index')
        plt.ylabel('Normalized Singular Value')
        plt.title('Singular Value Decay')
        plt.legend()
        plt.grid(True)

        # Plot log-log
        plt.subplot(122)
        plt.loglog(x, s_norm, 'ko-', label='Singular values')
        plt.loglog(x, exp_fit, 'r-', label=f'Exp fit (R²={r2_exp:.3f})')
        plt.loglog(x, power_fit, 'b-', label=f'Power fit (R²={r2_power:.3f})')
        plt.xlabel('Index')
        plt.ylabel('Normalized Singular Value')
        plt.title('Log-Log Plot of Singular Value Decay')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return popt_exp[1], r2_exp


def find_singular_value_entropy(adj_matrix, plot=True):
 
    # Compute SVD
    _, s, _ = svd(adj_matrix)

    # Normalize singular values (by sum for probability interpretation)
    s_norm = s / np.sum(s)

    # Compute Shannon entropy
    shannon_entropy = -np.sum(s_norm * np.log2(s_norm + 1e-12))

    # Compute participation ratio (effective dimensionality)
    participation_ratio = 1 / np.sum(s_norm ** 2)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot normalized singular value distribution
        x = np.arange(1, len(s) + 1)
        ax1.bar(x, s_norm, alpha=0.6)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Normalized Singular Value')
        ax1.set_title(f'Singular Value Distribution\nShannon Entropy: {shannon_entropy:.2f}')
        ax1.grid(True)

        # Plot cumulative distribution
        ax2.plot(x, np.cumsum(s_norm), 'b-')
        ax2.set_xlabel('Number of Singular Values')
        ax2.set_ylabel('Cumulative Sum')
        ax2.set_title(f'Cumulative Distribution\nParticipation Ratio: {participation_ratio:.2f}')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return shannon_entropy, participation_ratio


def generate_random_directed_graph(n, out_degree, seed=None):
    # Initialize directed graph
    if seed is not None:
        np.random.seed(seed)

    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(range(n))

    # Add edges with out-degree constraint
    for i in range(n):
        # Randomly select `out_degree` unique targets
        targets = np.random.choice(np.delete(np.arange(n), i), size=out_degree, replace=False)
        directed_graph.add_edges_from((i, t) for t in targets)

    # Convert to adjacency matrix
    adjacency_matrix = nx.to_numpy_array(directed_graph, dtype=int)
    return adjacency_matrix


def in_and_out_degree(W):
    out_degree = np.sum(W, axis=1)
    in_degree = np.sum(W, axis=0)
    return out_degree, in_degree


def compute_laplacian_matrix(w):
    return laplacian(w)


def generate_random_graph(num_nodes, density, rand_seed=None):
    np.random.seed(rand_seed)
    w = np.random.rand(num_nodes, num_nodes) < density
    np.fill_diagonal(w,0)
    return w


def generate_ER_graph(num_nodes, density, rand_seed=42):
    G = nx.fast_gnp_random_graph(num_nodes, density, seed=rand_seed, directed=True)
    w = convert_graph_to_adjacency_matrix(G)
    np.fill_diagonal(w, 0)
    return w


def generate_modular_graph(sz, pr, rand_seed=42):
    w = nx.adjacency_matrix(nx.stochastic_block_model(sz, pr, seed=rand_seed)).toarray()
    np.fill_diagonal(w, 0)
    return w


def FloydWarshall_Numba(adjmatrix, weighted_dist=False):
    @jit(nopython=True)
    def FW_Undirected(distmatrix):
        """The Floyd-Warshall algorithm for undirected networks
        """
        N = len(distmatrix)
        for k in range(N):
            for i in range(N):
                for j in range(i, N):
                    d = distmatrix[i, k] + distmatrix[k, j]
                    if distmatrix[i, j] > d:
                        distmatrix[i, j] = d
                        distmatrix[j, i] = d

    @jit(nopython=True)
    def FW_Directed(distmatrix):
        N = len(distmatrix)
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    d = distmatrix[i, k] + distmatrix[k, j]
                    if distmatrix[i, j] > d:
                        distmatrix[i, j] = d

    if weighted_dist:
        distmatrix = np.where(adjmatrix == 0, np.inf, adjmatrix)
    else:
        distmatrix = np.where(adjmatrix == 0, np.inf, 1)

    # 1.2) Find out whether the network is directed or undirected
    recip = compute_reciprocity_weighted(adjmatrix)
    if recip == 1.0:
        FW_Undirected(distmatrix)
    else:
        FW_Directed(distmatrix)

    return distmatrix


def compute_modularity_index(adj_mat):
    G = convert_adjacency_matrix_to_graph(adj_mat)
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)

    return modularity


def FloydWarshal_dir_weighted(w):
    INF = np.inf
    V = w.shape[0]
    dist = w.copy()
    dist = np.where(w == 0, INF, w)
    for k in range(V):
        # Use broadcasting to calculate the shortest path from i to j via k
        dist = np.minimum(dist, dist[:, k].reshape(-1, 1) + dist[k, :])

    return dist

def weight_to_length(weight_matrix):
    weight_matrix = np.array(weight_matrix, dtype=float)
    max_weight = np.max(weight_matrix)
    if max_weight == 0:
        raise ValueError("Maximum weight is zero, cannot perform weight-to-length remapping.")
    normalized_matrix = weight_matrix / max_weight
    inverse_matrix = np.copy(normalized_matrix)
    print('min and max: ', inverse_matrix.min(), inverse_matrix.max())
    non_zero_mask = normalized_matrix != 0
    inverse_matrix[non_zero_mask] = 1 / (normalized_matrix[non_zero_mask] + 1)
    print('min and max: ', inverse_matrix.min(), inverse_matrix.max())
    # length_matrix = np.log10((weight_matrix / max_weight) + 1)
    return inverse_matrix

def BellmanFord(w):
    w = weight_to_length(w)
    # if np.any(w < 0):
    #     raise ValueError("Weight matrix contains negative weights.")
    G = convert_adjacency_matrix_to_graph(w)
    try:
        length = dict(nx.all_pairs_bellman_ford_path_length(G, weight='weight'))
        dist = np.zeros(w.shape)
        np.fill_diagonal(dist, 0)
        dist[w == 0] = np.inf
        for source in range(w.shape[0]):
            for target in range(w.shape[0]):
                try:
                    # Attempt to retrieve the length value
                    dist[source, target] = length[source][target]
                except KeyError:
                    # If the key is not found, assign np.inf or np.nan
                    dist[source, target] = np.inf
    except nx.NetworkXUnbounded:
        # Handle the case where a negative weight cycle is detected
        print("Negative weight cycle detected. Returning a zero matrix.")
        # Return a zero matrix in case of negative weight cycle
        dist = np.zeros(w.shape)

    return dist


def departure_from_normality(M):
    M = np.array(M)
    # Calculate the Frobenius norm of M
    norm_F = norm(M, 'fro')
    # Compute the eigenvalues of M
    eigenvalues = np.linalg.eigvals(M)
    # Compute the sum of the squares of the eigenvalues
    sum_squares_eigenvalues = np.sum(np.abs(eigenvalues)**2)
    # Compute the departure from normality
    d_F = np.sqrt(norm_F**2 - sum_squares_eigenvalues)
    return d_F/norm_F


def generate_smallworld_graph(num_nodes, rewiring_prob=0.4, num_neighbors=10, rand_seed=None):
    undirected_graph = nx.watts_strogatz_graph(num_nodes, num_neighbors, rewiring_prob, seed=rand_seed)
    return nx.to_numpy_array(undirected_graph, dtype=int)


def generate_hierarchical_modular(mx_lvl, E, sz_cl, desired_links=None, rand_seed=None):
    """
    Generate a hierarchical modular network with a specified number of links.
    The topology varies with random seed while maintaining the link count.

    Parameters:
    -----------
    mx_lvl : int
        Maximum hierarchical level
    E : float
        Base for exponential probability scaling
    sz_cl : int
        Size of clusters
    desired_links : int, optional
        Exact number of links desired in the network
    rand_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    w : numpy.ndarray
        Adjacency matrix of the network
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # Initial template
    t = np.ones((2, 2))
    n = 2 ** mx_lvl
    sz_cl -= 1

    # Build hierarchical structure
    for lvl in range(1, mx_lvl):
        s = 2 ** (lvl + 1)
        CIJ = np.ones((s, s))
        grp1 = range(int(s / 2))
        grp2 = range(int(s / 2), s)
        ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
        ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
        CIJ.flat[ix1] = t
        CIJ.flat[ix2] = t
        CIJ += 1
        t = CIJ.copy()

    CIJ -= (np.ones((s, s)) + mx_lvl * np.eye(s))

    # Calculate connection probabilities
    ee = mx_lvl - CIJ - sz_cl
    ee = (ee > 0) * ee
    prob = (1 / E ** ee) * (np.ones((s, s)))

    if desired_links is None:
        # Original behavior - random connections based on probability
        CIJ = (prob > np.random.random((n, n)))
    else:
        # Use probabilities as weights for random sampling
        flat_prob = prob.flatten()

        # Create array of all possible edge indices
        row_indices, col_indices = np.meshgrid(np.arange(n), np.arange(n))
        all_edges = list(zip(row_indices.flatten(), col_indices.flatten()))

        # Randomly sample edges based on probabilities
        selected_edges = np.random.choice(
            len(all_edges),
            size=desired_links,
            replace=False,
            p=flat_prob / flat_prob.sum()
        )

        # Create empty adjacency matrix
        CIJ = np.zeros((n, n), dtype=bool)

        # Fill in the randomly selected edges
        for idx in selected_edges:
            i, j = all_edges[idx]
            CIJ[i, j] = True

    w = np.array(CIJ, dtype=int)
    return w


def generate_hierarchical_core_periphery(mx_lvl, E, sz_cl, desired_links=None, rand_seed=None):
    """
    Generate a hierarchical core-periphery network with a specified number of links.
    The topology varies with random seed while maintaining the link count.

    Parameters:
    -----------
    mx_lvl : int
        Maximum hierarchical level
    E : float
        Base for exponential probability scaling
    sz_cl : int
        Size of clusters
    desired_links : int, optional
        Exact number of links desired in the network
    rand_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    w : numpy.ndarray
        Adjacency matrix of the network
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # Initial template
    t = np.ones((2, 2))
    np.fill_diagonal(t, [2, 0])  # Core-periphery structure
    n = 2 ** mx_lvl

    # Build hierarchical structure
    for lvl in range(1, mx_lvl):
        s = 2 ** (lvl + 1)
        # Create block matrix with core-periphery structure
        CIJ = np.block([[t, np.ones((int(s / 2), int(s / 2)))],
                        [np.ones((int(s / 2), int(s / 2))), np.zeros((int(s / 2), int(s / 2)))]])
        CIJ += 1
        t = CIJ.copy()

    CIJ -= (np.ones((s, s)) + mx_lvl * np.eye(s))

    # Calculate connection probabilities
    ee = mx_lvl - CIJ - sz_cl
    ee = (ee > 0) * ee
    prob = (1 / E ** ee) * (np.ones((s, s)))

    if desired_links is None:
        # Original behavior - random connections based on probability
        CIJ = (prob > np.random.random((n, n)))
    else:
        # Use probabilities as weights for random sampling
        flat_prob = prob.flatten()

        # Create array of all possible edge indices
        row_indices, col_indices = np.meshgrid(np.arange(n), np.arange(n))
        all_edges = list(zip(row_indices.flatten(), col_indices.flatten()))

        # Randomly sample edges based on probabilities
        selected_edges = np.random.choice(
            len(all_edges),
            size=desired_links,
            replace=False,
            p=flat_prob / flat_prob.sum()
        )

        # Create empty adjacency matrix
        CIJ = np.zeros((n, n), dtype=bool)

        # Fill in the randomly selected edges
        for idx in selected_edges:
            i, j = all_edges[idx]
            CIJ[i, j] = True

    w = np.array(CIJ, dtype=int)
    return w


def generate_hierarchical_modular_core_periphery(mx_lvl, E, sz_cl, desired_links=None, rand_seed=None):
    """
    Generate a hierarchical modular core-periphery network with a specified number of links.
    The topology varies with random seed while maintaining the link count.

    Parameters:
    -----------
    mx_lvl : int
        Maximum hierarchical level
    E : float
        Base for exponential probability scaling
    sz_cl : int
        Size of clusters
    desired_links : int, optional
        Exact number of links desired in the network
    rand_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    w : numpy.ndarray
        Adjacency matrix of the network
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # Initial template
    t = np.ones((2, 2))
    n = 2 ** mx_lvl
    sz_cl -= 1

    # Build hierarchical structure
    for lvl in range(1, mx_lvl):
        s = 2 ** (lvl + 1)
        CIJ = np.ones((s, s))

        grp1 = range(int(s / 2))
        grp2 = range(int(s / 2), s)
        ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
        ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()

        # Set different connection strengths for core and periphery
        CIJ.flat[ix1] = t
        CIJ.flat[ix2] = 0.9 * t
        CIJ += 1
        t = CIJ.copy()

    CIJ -= (np.ones((s, s)) + mx_lvl * np.eye(s))

    # Calculate connection probabilities with modified scaling
    ee = mx_lvl - CIJ - sz_cl
    ee = (ee > 0) * ee + np.ones((s, s))
    prob = 1 / E ** ee

    if desired_links is None:
        # Original behavior - random connections based on probability
        CIJ = (prob > np.random.random((n, n)))
    else:
        # Use probabilities as weights for random sampling
        flat_prob = prob.flatten()

        # Create array of all possible edge indices
        row_indices, col_indices = np.meshgrid(np.arange(n), np.arange(n))
        all_edges = list(zip(row_indices.flatten(), col_indices.flatten()))

        # Randomly sample edges based on probabilities
        selected_edges = np.random.choice(
            len(all_edges),
            size=desired_links,
            replace=False,
            p=flat_prob / flat_prob.sum()
        )

        # Create empty adjacency matrix
        CIJ = np.zeros((n, n), dtype=bool)

        # Fill in the randomly selected edges
        for idx in selected_edges:
            i, j = all_edges[idx]
            CIJ[i, j] = True

    w = np.array(CIJ, dtype=int)
    return w



def generate_input(signal_type='random', num_samples=1000, input_dim=1, **kwargs):
    """
    Generate input signals for reservoir computing.

    Parameters:
    -----------
    signal_type : str
        Type of signal to generate:
        - 'random': Gaussian random signal
        - 'sine': Sinusoidal signal
        - 'square': Square wave
        - 'mixed': Sum of multiple sine waves
    num_samples : int
        Length of the signal
    input_dim : int
        Number of input dimensions
    kwargs : dict
        Additional parameters:
        - 'freq': Frequency for sine/square waves (default: 0.1)
        - 'amplitude': Signal amplitude (default: 1.0)
        - 'noise': Noise level to add (default: 0.0)

    Returns:
    --------
    signal : ndarray
        Generated signal of shape (num_samples, input_dim)
    """
    # Get parameters from kwargs
    freq = kwargs.get('freq', 0.1)
    amplitude = kwargs.get('amplitude', 1.0)
    noise = kwargs.get('noise', 0.0)

    # Time vector
    t = np.linspace(0, num_samples/10, num_samples)

    # Initialize output array
    signal = np.zeros((num_samples, input_dim))

    for i in range(input_dim):
        if signal_type == 'random':
            # Generate random Gaussian signal
            signal[:, i] = generate_iid_signal(num_samples, 'gaussian', mu=0, sigma=1)

        elif signal_type == 'sine':
            # Generate sine wave
            signal[:, i] = amplitude * np.sin(2 * np.pi * freq * t)

        elif signal_type == 'square':
            # Generate square wave
            signal[:, i] = amplitude * np.sign(np.sin(2 * np.pi * freq * t))

        elif signal_type == 'mixed':
            # Sum of three sine waves with different frequencies
            f1, f2, f3 = freq, 2*freq, 3*freq
            signal[:, i] = amplitude * (np.sin(2 * np.pi * f1 * t) +
                                      0.5 * np.sin(2 * np.pi * f2 * t) +
                                      0.25 * np.sin(2 * np.pi * f3 * t))
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        # Add noise if specified
        if noise > 0:
            signal[:, i] += noise * np.random.randn(num_samples)

    return signal

def check_strongly_connected(adj_matrix):
    """
    Check if the directed network is strongly connected.
    """
    n_components, labels = connected_components(
        adj_matrix,
        directed=True#,
        # connection='strong'
    )
    return n_components == 1


def generate_random_asymmetric_matrix(n, avg_out_degree, rand_seed=None):
    """
    Generates an asymmetric random matrix with a given average out-degree.

    Parameters:
        n (int): Number of nodes (matrix dimension).
        avg_out_degree (float): Desired average out-degree.
        weight_range (tuple): Range of weights for non-zero entries (default: (0, 1)).

    Returns:
        np.ndarray: A random asymmetric matrix.
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)
    total_connections = int(n * avg_out_degree)

    # Initialize an empty matrix
    matrix = np.zeros((n, n), dtype=int)

    # Get all possible non-diagonal indices for an asymmetric matrix
    all_indices = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Randomly select indices for connections
    selected_indices = np.random.choice(len(all_indices), size=total_connections, replace=False)

    # Assign 1s to the selected indices
    for idx in selected_indices:
        i, j = all_indices[idx]
        matrix[i, j] = 1

    return matrix


def generate_temporal_pattern(length, n_patterns):
    patterns = []
    for _ in range(n_patterns):
        # Create sinusoidal pattern with random frequency and phase
        t = np.linspace(0, 10*np.pi, length)
        freq = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2*np.pi)
        pattern = np.sin(freq * t + phase)
        patterns.append(pattern)
    return np.array(patterns)

def generate_sequential_pattern(length, n_patterns):
    patterns = []
    for _ in range(n_patterns):
        # Create sequence with specific transitions
        pattern = np.zeros(length)
        current = np.random.choice([-1, 1])
        for i in range(length):
            pattern[i] = current
            if np.random.random() < 0.2:  # 20% chance to switch
                current *= -1
        patterns.append(pattern)
    return np.array(patterns)

def generate_hierarchical_pattern(length, n_patterns, levels=3):
    patterns = []
    for _ in range(n_patterns):
        # Create pattern with different frequencies
        pattern = np.zeros(length)
        for level in range(levels):
            freq = 2**(level+1)
            pattern += np.sin(freq * np.linspace(0, 10*np.pi, length))
        patterns.append(pattern/levels)  # Normalize
    return np.array(patterns)

# Function to enforce connectivity
def enforce_connectivity(M):
    # Get dimensions
    _, _, i_dim, t_dim, j_dim = M.shape

    # Loop over all i and j combinations
    for i in range(i_dim):
        for j in range(j_dim):
            # Keep track of the most recent connected matrix
            last_connected = None

            for t in range(t_dim):
                current_matrix = M[:, :, i, t, j]

                if not check_strongly_connected(current_matrix):

                    if last_connected is not None:
                        # Replace with the most recent connected matrix
                        M[:, :, i, t, j] = last_connected
                    else:
                        M[:, :, i, t, j] = M[:, :, i, t, j]
                        # raise ValueError(
                        #     f"No previously connected matrix found for i={i}, j={j}, t={t}. Check input data!"
                        # )
                else:
                    # Update the last connected matrix
                    last_connected = current_matrix

    return M


import numpy as np


def compute_eigenvalue_gap(matrix):
    """
    Compute the gap between the largest and second largest eigenvalues (in magnitude)
    """
    # Compute eigenvalues
    eigenvalues = LA.eigvals(matrix)
    # Sort eigenvalues by magnitude in descending order
    sorted_eigenvalues = sorted(abs(eigenvalues), reverse=True)

    # If matrix has at least 2 eigenvalues, compute gap
    if len(sorted_eigenvalues) >= 2:
        gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    else:
        gap = sorted_eigenvalues[0]  # If only one eigenvalue exists

    return gap





def generate_iid_signal(n, distribution='gaussian', **params):
    """
    Generate an i.i.d (independent and identically distributed) signal of length n.

    Parameters:
    -----------
    n : int
        Length of the signal to generate
    distribution : str, optional
        Type of distribution to use (default: 'gaussian')
        Options: 'gaussian', 'uniform', 'bernoulli', 'exponential'
    **params : dict
        Additional parameters for the specific distribution

    Returns:
    --------
    numpy.ndarray
        Array of length n containing the i.i.d samples

    Examples:
    --------
    # Generate Gaussian signal
    x = generate_iid_signal(1000, 'gaussian', mu=0, sigma=1)

    # Generate Uniform signal
    x = generate_iid_signal(1000, 'uniform', low=-1, high=1)

    # Generate Bernoulli signal
    x = generate_iid_signal(1000, 'bernoulli', p=0.5)
    """

    if distribution.lower() == 'gaussian':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        signal = np.random.normal(mu, sigma, n)

    elif distribution.lower() == 'uniform':
        low = params.get('low', -1)
        high = params.get('high', 1)
        signal = np.random.uniform(low, high, n)

    elif distribution.lower() == 'bernoulli':
        p = params.get('p', 0.5)
        signal = np.random.binomial(1, p, n)

    elif distribution.lower() == 'exponential':
        scale = params.get('scale', 1.0)
        signal = np.random.exponential(scale, n)

    else:
        raise ValueError(f"Distribution '{distribution}' not supported")

    return signal