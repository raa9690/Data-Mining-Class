import pandas  as pd
import numpy   as np
import scipy.signal
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics

CSV_PATH = "../DataMining_Data/HW_CLUSTERING_SHOPPING_CART_v2225a.csv"

def open_data_file():
    # read the file
    df = pd.read_csv(CSV_PATH, index_col=False)
    del df['ID']
    return df

# generates a matrix of cross correlation coefficients for all columns in a dataframe, returns nxn numpy array
def cross_correlate(df):
    # create the nxn numpy array to fill in the values of the cross correlation matrix
    cross_correlate_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for row in range(df.shape[1]):
        for col in range(df.shape[1]):
            # struggled to find a good library that worked well with dataframes for the cross corelations coefficient,
            # so I just used numpy corrcoef to get the covariance matrix, which has the value of the cross correlation
            # coefficient
            cov_matrix = np.cov(df.iloc[:,col].T.to_numpy(), df.iloc[:,row].T.to_numpy())
            cross_corr_coef = cov_matrix[0][1] / ( math.sqrt(cov_matrix[0][0]) * math.sqrt(cov_matrix[1][1]) )
            cross_correlate_matrix[col][row] = round(cross_corr_coef,2)
    return cross_correlate_matrix

# outputs info used to answer the report
def output_report(cross_matrix):
    # print cross correlation between chips and cereal
    print("CrossCoef Chips and Cereal: ", cross_matrix[9][3])
    # print indexes of the largest correlation for fish
    # get a copy of the col, sort and get the second-largest value (largest will always be with itself)
    # then find all the indexes with that value
    sorted_cross_matrix_row = cross_matrix[13].copy()
    sorted_cross_matrix_row.sort()
    indices = []
    # loop through all indices and find where they are equal to the second-largest cross correlation
    for i in range(cross_matrix.shape[0]):
        if (cross_matrix[13][i] == sorted_cross_matrix_row[-2]):
            indices.append(i)
    print("Indices of item that fish is most strongly correlated with: ",indices)
    # print indexes of the largest correlation for veggies
    # get a copy of the col, sort and get the second-largest value (largest will always be with itself)
    # then find all the indexes with that value
    sorted_cross_matrix_row = cross_matrix[2].copy()
    sorted_cross_matrix_row.sort()
    indices = []
    # loop through all indices and find where they are equal to the second-largest cross correlation
    for i in range(cross_matrix.shape[0]):
        if (cross_matrix[2][i] == sorted_cross_matrix_row[-2]):
            indices.append(i)
    print("Indices of item that veggies is most strongly correlated with: ",indices)
    print("Cross Correlation between milk and cereal: ",cross_matrix[0][3])
    print("================================================")
    #
    for i in range(cross_matrix[0].size):
        print(i)
        sorted_cross_matrix_row = cross_matrix[i].copy()
        sorted_cross_matrix_row.sort()
        positive_indices = []
        # loop through all indices and find where they are equal to the second-largest cross correlation
        for j in range(cross_matrix.shape[0]):
            if (cross_matrix[i][j] == sorted_cross_matrix_row[-2]):
                positive_indices.append(j)

        negative_indices = []
        # loop through all indices and find where they are equal to the second-largest cross correlation
        for j in range(cross_matrix.shape[0]):
            if (cross_matrix[i][j] == sorted_cross_matrix_row[0]):
                negative_indices.append(j)
        # see what attributes have the smallest and second-largest coefficient
        print("Smallest coefficient: ",sorted_cross_matrix_row[0])
        print("Largest coefficient: ",sorted_cross_matrix_row[-2])
        print("Index ",i," has it's highest positive correlation with indices: ",positive_indices)
        print("Index ",i," has it's highest negative correlation with indices: ",negative_indices)
        # getting the sum of the absolute values to see what attributes/features have the lowest total cross-correlation
        print("Sum of coefficients: ",(np.absolute(cross_matrix[i].copy())).sum())

def main():
    df = open_data_file()
    cross_matrix = cross_correlate(df)
    output_report(cross_matrix)


main()