#%%
import pandas  as pd
import numpy   as np
import scipy.signal
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics
from sklearn.cluster import KMeans

CSV_PATH = "../../DataMining_Data/HW_CLUSTERING_SHOPPING_CART_v2225a1.csv"
CUMILATIVE_EIGEN_VAL_PLOT_FILE = "Cumilative_Sum_Sorted_EigenVal.png"
PROJECTION_PLOT = "Eigen_Vector_Projection_Plot.png"
K_MEANS_PCA_PLOT = "K_Means_PCA_First_Two_Eigenvectors_Plot.png"

# reads in a csv file from the path CSV_PATH
def read_file():
    # read the file
    df = pd.read_csv(CSV_PATH, index_col=False)
    del df['ID']
    return df

# function to get the covariance matrix
def get_cov_matrix(df):
    cov_matrix = df.cov()
    return cov_matrix

# from a covariance matrix, get the eigenvectors and eigenvalues
def get_eigen_vectors_values(cov_matrix):
    # huh, example I found to make sure I was doing this right happened to name everything the same way I did
    eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
    return eigen_vals,eigen_vecs

# function that performs insertion sort and returns the sorted array of numbers, and an array representing the indexes
# of the original array and where they have been shifted to
def insertion_sort(array_of_numbers):
    # get an array representing the indices of the array of numbers
    # as we sort the array, we'll also be shifting the indices, this will make it easier to keep track of what
    # eigen vector the given eigen value is associated with
    index_array = np.array(range(array_of_numbers.shape[0]))
    print("======================")
    print(index_array)
    print("======================")
    print(array_of_numbers)


    absolute_total = 0 # might as well use this loop to get the absolute total
    for i in range(array_of_numbers.shape[0]):
        for j in range(i,array_of_numbers.shape[0]):
            # search the indices to the right of the current index
            # if the absolute value is greater than what's in the current index, swap the values in both the index array
            # and the array_of_numbers
            if np.absolute(array_of_numbers[i]) < np.absolute(array_of_numbers[j]):
                temp_value = array_of_numbers[i]
                array_of_numbers[i] = array_of_numbers[j]
                array_of_numbers[j] = temp_value
                temp_index = index_array[i]
                index_array[i] = index_array[j]
                index_array[j] = temp_index
        absolute_total += np.absolute(array_of_numbers[i])
    print("======================")
    return array_of_numbers, index_array, absolute_total


# just normalizes the eigenvals
def normalize_eigen_vals(sorted_eigen_vals, absolute_total):
    normalized_sorted_eigen_vals = sorted_eigen_vals / absolute_total
    print(normalized_sorted_eigen_vals)
    return normalized_sorted_eigen_vals


def first_three_eig_vec_info_print(normalized_sorted_eigen_vals, eigen_vecs, adjusted_indices):
    cumilative_sum_sorted_eigen_vals = pd.DataFrame.cumsum(pd.DataFrame(normalized_sorted_eigen_vals))
    cumilative_sum_sorted_eigen_vals.plot()
    plt.savefig(CUMILATIVE_EIGEN_VAL_PLOT_FILE)
    print("Eigen Vector 1: ",eigen_vecs[adjusted_indices[0]])
    print("Eigen Vector 2: ",eigen_vecs[adjusted_indices[1]])
    print("Eigen Vector 3: ",eigen_vecs[adjusted_indices[2]])
    sum_of_attr_first_3_eig_vec = np.absolute(eigen_vecs[adjusted_indices[0]]) \
                                  + np.absolute(eigen_vecs[adjusted_indices[1]]) \
                                  + np.absolute(eigen_vecs[adjusted_indices[2]])
    print("Sum of first 3 Eigen Vectors: ", sum_of_attr_first_3_eig_vec)
    sum_of_attr_eig_vec_2_and_3 = np.absolute(eigen_vecs[adjusted_indices[1]]) \
                                  + np.absolute(eigen_vecs[adjusted_indices[2]])
    print("Sum of second and third Eigen Vectors: ", sum_of_attr_eig_vec_2_and_3)


def eigen_vec_projection(df, eigen_vecs, normalized_sorted_eigen_vals, adjusted_indices, ):
    # matrix of the first two eigenvectors, in order of the largest absolute eigenvalue
    eigen_vecs_to_use = np.array([eigen_vecs[adjusted_indices[1]], eigen_vecs[adjusted_indices[2]]])
    # get the dot product to project the data onto the eigen vectors
    projection_result = np.dot(df, eigen_vecs_to_use.T)
    return projection_result, eigen_vecs_to_use


def plot_eigen_vec_projection(projection_result):
    projection_df = pd.DataFrame(projection_result)
    projection_df.plot(0,1,kind='scatter', figsize=(10,10))
    sns.scatterplot(data=projection_df,x=0,y=1)
    plt.savefig(PROJECTION_PLOT)
    plt.clf()
    projection_df.plot(0, 0, kind='scatter')
    plt.savefig("Projection_1_"+PROJECTION_PLOT)
    plt.clf()
    projection_df.plot(1, 1, kind='scatter')
    plt.savefig("Projection_2_"+PROJECTION_PLOT)
    plt.clf()


def plot_k_means_center(centers)

def plot_k_means_center_vectors_PCA(centers):
    colors = ['red',
              'blue',
              'green',
              'orange',
              'violet',
              'yellow',
              'magenta',
              'teal',
              'lime',
              'brown']
    # using this as a base point for the start of the vector
    numpy_zeroes = np.zeros(centers.shape[0])
    # adds the vectors on top of the clustering plot
    plt.quiver(numpy_zeroes,numpy_zeroes,centers[:,0], centers[:,1],
               color=colors[:centers.shape[0]], scale=1, scale_units='xy', edgecolor='black', linewidth=1)
    plt.show()


def k_means(projection_result, number_of_clusters):
    if (number_of_clusters > 10):
        print("Maximum number of clusters is 10, setting the number of clusters to 10")
        number_of_clusters = 10
    projection_result_df = pd.DataFrame(projection_result)
    k_means = KMeans(n_clusters=number_of_clusters)
    label = k_means.fit_predict(projection_result_df)

    # filter rows of original data
    # allows for a maximum of 10 clusters
    filtered_label0 = projection_result_df[label == 0]
    filtered_label1 = projection_result_df[label == 1]
    filtered_label2 = projection_result_df[label == 2]
    filtered_label3 = projection_result_df[label == 3]
    filtered_label4 = projection_result_df[label == 4]
    filtered_label5 = projection_result_df[label == 5]
    filtered_label6 = projection_result_df[label == 6]
    filtered_label7 = projection_result_df[label == 7]
    filtered_label8 = projection_result_df[label == 8]
    filtered_label9 = projection_result_df[label == 9]

    # Plotting the results
    # Allows for plotting of a maximum of 10 clusters
    plt.figure(figsize=(10,10))
    plt.scatter(x=filtered_label0[0], y=filtered_label0[1], color='red')
    plt.scatter(x=filtered_label1[0], y=filtered_label1[1], color='blue')
    plt.scatter(x=filtered_label2[0], y=filtered_label2[1], color='green')
    plt.scatter(x=filtered_label3[0], y=filtered_label3[1], color='orange')
    plt.scatter(x=filtered_label4[0], y=filtered_label4[1], color='violet')
    plt.scatter(x=filtered_label5[0], y=filtered_label5[1], color='yellow')
    plt.scatter(x=filtered_label6[0], y=filtered_label6[1], color='magenta')
    plt.scatter(x=filtered_label7[0], y=filtered_label7[1], color='teal')
    plt.scatter(x=filtered_label8[0], y=filtered_label8[1], color='lime')
    plt.scatter(x=filtered_label9[0], y=filtered_label9[1], color='brown')
    plt.savefig(K_MEANS_PCA_PLOT)

    # get the centers
    centers = k_means.cluster_centers_
    # plot the centers as vectors from point 0,0
    plot_k_means_center_vectors_PCA(centers)
    
    return centers


def reprojection_print(centers, eigenvectors_used):
    reprojection = np.dot(centers, eigenvectors_used)
    print(reprojection)
    print(reprojection.shape)


def main():
    df = read_file()

    cov_matrix = get_cov_matrix(df)

    eigen_vals, eigen_vecs = get_eigen_vectors_values(cov_matrix)

    sorted_eigen_vals, adjusted_indices, absolute_total = insertion_sort(eigen_vals)

    normalized_sorted_eigen_vals = normalize_eigen_vals(sorted_eigen_vals,absolute_total)

    first_three_eig_vec_info_print(normalized_sorted_eigen_vals, eigen_vecs, adjusted_indices)

    projection_result, eigenvectors_used = eigen_vec_projection(df, eigen_vecs, normalized_sorted_eigen_vals, adjusted_indices)

    plot_eigen_vec_projection(projection_result)

    centers = k_means(projection_result, 3)

    reprojection_print(centers, eigenvectors_used)




main()


#%%
