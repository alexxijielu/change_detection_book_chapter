'''Given the profiled features for two screens, calculate the protein change profiles between the two screens.
Author: Alex Lu
Email: alexlu@cs.toronto.edu
Last Updated: May 23th, 2018

Copyright (C) 2018 Alex Lu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see
<https://www.gnu.org/licenses/>.'''

from .util import openGeneMatrix
from .util import packageGeneMatrix
from sklearn import preprocessing
import numpy as np
import sklearn.metrics.pairwise as skdist

def filter_matrices (reference, condition, scale=True, drop_sparse=False):
    '''Preprocessing operation - calculates the intersection of proteins between the reference and the
    condition, and sorts them so that they're in the same order.'''
    ref_headers, ref_genelist, ref_genematrix = openGeneMatrix(reference)
    cond_headers, cond_genelist, cond_genematrix = openGeneMatrix(condition)

    if drop_sparse:
        sparse_indices = []
        sparsity_limit = 0.30

        for column in range (0, cond_genematrix.shape[1]):
            current_column = cond_genematrix[:, column]
            if (len(current_column) - np.count_nonzero(current_column)) / len(current_column) > sparsity_limit:
                sparse_indices.append(column)

        for column in range (0, ref_genematrix.shape[1]):
            current_column = ref_genematrix[:, column]
            if (len(current_column) - np.count_nonzero(current_column)) / len(current_column) > sparsity_limit:
                sparse_indices.append(column)

        sparse_indices = np.array(sparse_indices)
        cond_genematrix = np.delete(cond_genematrix, sparse_indices, axis=1)
        ref_genematrix = np.delete(ref_genematrix, sparse_indices, axis=1)
        cond_headers = np.delete(cond_headers, sparse_indices)
        ref_headers = np.delete(ref_headers, sparse_indices)

    if scale:
        ref_genematrix = preprocessing.scale(ref_genematrix)
        cond_genematrix = preprocessing.scale(cond_genematrix)

    sorted_ref = []
    sorted_cond = []
    # Get the intersection of the list
    # Sometimes there can be duplicate proteins, so we'll just take the first occurrence if there is
    intersect = np.intersect1d(ref_genelist, cond_genelist)
    for protein in intersect:
        ref_index = np.where(ref_genelist == protein)[0][0]
        cond_index = np.where(cond_genelist == protein)[0][0]
        sorted_ref.append(ref_genematrix[ref_index])
        sorted_cond.append(cond_genematrix[cond_index])

    sorted_ref = np.array(sorted_ref)
    sorted_cond = np.array(sorted_cond)

    return intersect, sorted_ref, sorted_cond, cond_headers

def subtract_matrices (sorted_ref, sorted_cond):
    '''Produce a subtracted matrix given two sorted matrices'''
    return np.subtract(sorted_ref, sorted_cond)

def modWeights(k, geneMatrix, distMatrix, metric='euclidean', verbose=True):
    '''Generates a mean and MAD vector for each protein in a protein feature matrix using the k closest genes using euclidean distance
    Inputs: k, wild-type protein feature matrix, change matrix
    Output: mean and MAD of k NN of proteins'''
    # Specify distance metric and get nearest neighbors
    dist = skdist.pairwise_distances(distMatrix, metric=metric)
    nearest = np.argsort(dist, axis=1)[:, 1:(k + 1)]

    # Calculate medians and MAD for each protein
    medians = np.zeros(geneMatrix.shape)
    MAD = np.zeros(geneMatrix.shape)
    for gene in range(0, nearest.shape[0]):
        neighbors = []
        for index in nearest[gene]:
            neighbors.append(geneMatrix[index])

        neighbors = np.array(neighbors)
        for feature in range(0, len(neighbors[0])):
            medians[gene][feature] = np.median(neighbors[:, feature])
            MAD[gene][feature] = np.median(abs(medians[gene][feature] - neighbors[:, feature]))
        if ((gene + 1) % 200 == 0 or (gene + 1) == geneMatrix.shape[0]) and verbose:
            print ("Calculated expectations for %d out of %d proteins." % ((gene + 1), geneMatrix.shape[0]))

    return medians, MAD

def calculateModZScores(geneMatrix, means, MAD):
    '''Calculates modified z-score vectors for each gene given mean and variance vectors of kNN neighbors
    Inputs: gene matrix and corresponding mean and variances of kNN neighbors
    Output: zscores'''
    zscores = np.zeros(geneMatrix.shape)
    for gene in range(0, geneMatrix.shape[0]):
        for feature in range(0, len(geneMatrix[gene])):
            if MAD[gene][feature] != 0:
                zscores[gene][feature] = (geneMatrix[gene][feature] - means[gene][feature]) / (MAD[gene][feature])
            else:
                # if we can't estimate the MAD, we'll just toss it out
                zscores[gene][feature] = 0
    return zscores

def remove_duplicates(genelist, subtracted, sorted_ref):
    '''Remove any duplicate features'''
    subtracted, indices = np.unique(subtracted, axis=0, return_index=True)
    sorted_ref = sorted_ref[indices]
    genelist = genelist[indices]
    return genelist, subtracted, sorted_ref

def knn_change_detection(reference, condition, output, k=50, verbose=True):
    if verbose:
        print ("Filtering to intersection of proteins in both datasets...")
    genelist, sorted_ref, sorted_cond, headers = filter_matrices(reference, condition, scale=True, drop_sparse=False)
    subtracted = subtract_matrices(sorted_ref, sorted_cond)
    if verbose:
        print ("Calculating z-scores...")
    means, variances = modWeights(k, subtracted, sorted_ref, verbose=verbose)
    zscores = calculateModZScores(subtracted, means, variances)
    if verbose:
        print ("Saving output...")
    packageGeneMatrix(output, headers, genelist, zscores)
    if verbose:
        print ("Done!")

if __name__ == '__main__':
    reference = r"C:\Users\lualex\PycharmProjects\change_detection_book_chapter\data\human_cell_lines\U-2-OS_features.tsv"
    condition = r"C:\Users\lualex\PycharmProjects\change_detection_book_chapter\data\human_cell_lines\A-431_features.tsv"
    output = r"C:\Users\lualex\PycharmProjects\change_detection_book_chapter\data\human_cell_lines\U-2-OS_A-431_change.tsv"
    k = 50

    print ("Calculating protein localization change profiles...")
    genelist, sorted_ref, sorted_cond, headers = filter_matrices(reference, condition, scale=True, drop_sparse=False)
    subtracted = subtract_matrices(sorted_ref, sorted_cond)
    means, variances = modWeights(k, subtracted, sorted_ref)
    zscores = calculateModZScores(subtracted, means, variances)

    print ("Done!")
    packageGeneMatrix(output, headers, genelist, sorted_cond)
    #packageGeneMatrix(r"C:\Users\alexi_000\Desktop\new_change_detection\visualization\RAP3_expectation_proteins.txt",
    #                  headers, genelist, means)