'''Utility files meant for opening and closing gene matrixes. First column is assumed to be the IDs,
first header is assumed to be the feature names, and rest of data is assumed to be data.

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

import csv
import numpy as np

def openGeneMatrix(fileName):
    '''Opens a gene matrix file and returns the feature labels, gene labels and gene matrix
    Input: Path of file to be opened (as a string)
    Output: feature labels, gene labels, and gene matrix'''
    file = open(fileName)
    list = csv.reader(file, delimiter='\t')
    matrix = np.array([row for row in list])

    genelist = matrix[1:, 0]
    headers = matrix[0, :]
    genematrix = matrix[1:, 1:]
    try:
        genematrix = genematrix.astype(np.float32)
    except:
        pass

    file.close()
    return headers, genelist, genematrix

def packageGeneMatrix(fileName, headers, genelist, genematrix):
    '''Combines the feature labels, gene labels and gene matrix and writes to output
    Input: Path of file to be written to, feature labels, gene labels, and gene matrix
    Output: Writes to path of file'''
    file = open(fileName, "w")
    file.truncate()

    file.write('\t'.join(headers) + '\n')
    for i in range(0, genematrix.shape[0]):
        file.write(genelist[i] + '\t')
        for j in genematrix[i][:-1]:
            file.write((str(j)) + '\t')
        file.write(str(genematrix[i][-1]))
        file.write('\n')

    print("Written to file " + fileName)
    file.close()