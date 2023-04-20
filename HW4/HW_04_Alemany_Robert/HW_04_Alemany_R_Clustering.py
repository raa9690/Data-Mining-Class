import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics
import argparse


#
#  My hacked variance function.
#  This is defined so that if an empty array is passed in,
#  the variance returned is still returns zero.
#
def my_var(an_array):
    if len(an_array) == 0:
        ans = 0;
    else:
        ans = np.var(an_array)
    return ans


#
#  COMPUTE THE MIXED VARIANCE OF THE DATA:
#  Return the best threshold to minimize the function.
#
def get_otsu(thresholds, the_data):
    # Initialize to BOGUS values:
    best_mixed_variance = float('inf')
    best_threshold = None
    threshindx = 0
    valid_thresholds = []
    all_mixed_variances = []
    for threshold_idx in range(0, len(thresholds), 1):
        threshold = thresholds[threshold_idx]

        # Form the left set and the right set:
        set_left = []
        set_rght = []
        count_left = 0
        count_rght = 0
        for data_value in the_data:
            if data_value <= threshold:
                set_left.append(data_value)
                count_left = count_left + 1
            else:
                set_rght.append(data_value)
                count_rght = count_rght + 1

        wt_left = count_left / (count_left + count_rght)
        wt_rght = count_rght / (count_left + count_rght)

        var_left = my_var(set_left)  # np.var( set_left )
        var_rght = my_var(set_rght)  # np.var( set_rght )

        mixed_variance = wt_left * var_left + wt_rght * var_rght
        # print("trying threshold = ", threshold, "mixed variance = ", mixed_variance)

        # this should be < but mine only does >
        if not math.isnan(mixed_variance):
            valid_thresholds.append(threshold)
            all_mixed_variances.append(mixed_variance)
            if mixed_variance <= best_mixed_variance:
                best_mixed_variance = mixed_variance
                best_threshold = threshold

    return best_threshold, valid_thresholds, all_mixed_variances


# making a change from the code provided so now the main is a function that is called by name (to use argparse)
def otsu_clustering(csv_path):
    # get data
    all_the_data = pd.read_csv(csv_path, index_col=False)

    # take a quick look at data
    all_the_data.head()

    the_data = all_the_data['D8']

    # set quant amount
    quantization = 1.0 / 32.0

    # get min and max
    minn = the_data.min()
    maxx = the_data.max()

    # set the range of thresholds based on min and max amount
    thresholds = np.arange(minn, maxx, quantization)

    best_thresh, new_threshs, all_variances = get_otsu(thresholds, the_data)

    print('best threshold = ', best_thresh)

    #
    # Create a basic figure showing the mixed variance in the legal ranges:
    #
    the_max_var = np.max(all_variances)
    the_min_var = np.min(all_variances)

    plt.plot(new_threshs, all_variances, linewidth=4);
    plt.plot([best_thresh, best_thresh], [0, the_max_var], 'r-', linewidth=2);
    plt.grid()
    plt.title(' Mixed Variance as a Function of Threshold ', fontsize=24)
    plt.xlabel(' Plant Length (inches) ', fontsize=20)
    plt.ylabel(' Mixed Variance for Threshold', fontsize=20)
    fig = plt.gcf()  # Get the current figure.
    fig.set_size_inches(11, 8.5)  # Size of legal piece of paper
    ax = plt.gca()  # Get the current axis.
    msg = '<-- Best Threshold=' + str(best_thresh) + ' inches'
    delta_dst = (the_max_var - the_min_var)
    dst_up = delta_dst / 20
    ax.text(best_thresh + 0.15, dst_up, msg, fontsize=16, backgroundcolor='w');
    plt.savefig('Dr_K__Otsu_Answer.jpg', dpi=100)
    plt.show()

    return best_thresh


def create_hist_clust(title_name, csv_path, threshold, quants_per_binsize):
    # get data
    all_the_data = pd.read_csv(csv_path, index_col=False)

    # take a quick look at data
    all_the_data.head()

    the_data_values = all_the_data['D8']

    # using a binsize of the quant of the data used (1/32 inches)
    bin_size = (1/32)*quants_per_binsize
    # get the max binCount that would include the last value using the binsize
    bin_count = int(np.ceil((the_data_values.max() - the_data_values.min())/bin_size))

    plt.hist(the_data_values, bins=bin_count)
    plt.title(title_name)
    plt.xlabel("Height of Plant in Inches")
    plt.ylabel("Count")

    # draw the threshold line
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label='Threshold')

    plt.savefig('Histogram_Clustering.jpg', dpi=100)


#
#
#  QUICK PYTHON PROGRAM TO: Compute Otsu's Method:
#
#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file_path', help='Required: file path to a csv file for dataset')
    args = parser.parse_args()
    csv_path = args.csv_file_path
    threshold = otsu_clustering(csv_path)
    # creating a histogram with a binsize of at least 4 * (1/32) inches
    create_hist_clust("Clustering of Plants", csv_path, threshold, 4)


if __name__ == "__main__":
    main()
else:
    print("this is NOT main")

#%%
