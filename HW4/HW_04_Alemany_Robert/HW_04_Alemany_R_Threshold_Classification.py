import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


# based off of code provided, the_data has been changed to be a 2d array representing data_value class pairs
# rather than checking variance, it now checks the total error count
def get_otsu(thresholds, the_data):
    # Initialize to BOGUS values:
    best_error_count = float('inf')
    best_threshold = None
    threshindx = 0
    valid_thresholds = []
    all_error_counts = []
    for threshold_idx in range(0, len(thresholds), 1):
        threshold = thresholds[threshold_idx]

        # count how many false positives + false negatives for all the data
        error_count = 0
        index_counter = 0  # small hack to let us iterate through the data_values, and get the class at that index
        for data_value in the_data[0]:
            if data_value <= threshold:
                if the_data[1].get(index_counter) != 1:
                    error_count += 1
            else:
                if the_data[1].get(index_counter) == 1:
                    error_count += 1
            index_counter += 1

        valid_thresholds.append(threshold)
        all_error_counts.append(error_count)
        if error_count <= best_error_count:
            best_error_count = error_count
            best_threshold = threshold

    return best_threshold, valid_thresholds, all_error_counts


def otsu_classification(csv_path):
    # get data
    all_the_data = pd.read_csv(csv_path, index_col=False)

    # take a quick look at data
    all_the_data.head()

    the_data_values = all_the_data['Day8']
    the_data_classes = all_the_data['Cls']
    the_data = [the_data_values, the_data_classes]

    # set quant amount
    quantization = 1.0 / 32.0

    # get min and max
    minn = the_data_values.min()
    maxx = the_data_values.max()

    # set the range of thresholds based on min and max amount
    thresholds = np.arange(minn, maxx, quantization)

    best_thresh, new_threshs, all_error_counts = get_otsu(thresholds, the_data)

    print('best threshold = ', best_thresh)

    #
    # Create a basic figure showing the mixed variance in the legal ranges:
    #
    the_max_err = np.max(all_error_counts)
    the_min_err = np.min(all_error_counts)

    plt.plot(new_threshs, all_error_counts, linewidth=4);
    plt.plot([best_thresh, best_thresh], [0, the_max_err], 'r-', linewidth=2);
    plt.grid()
    plt.title(' Error Count as a Function of Threshold ', fontsize=24)
    plt.xlabel(' Plant Length (inches) ', fontsize=20)
    plt.ylabel(' Error Count for Threshold', fontsize=20)
    fig = plt.gcf()  # Get the current figure.
    fig.set_size_inches(11, 8.5)  # Size of legal piece of paper
    ax = plt.gca()  # Get the current axis.
    msg = '<-- Best Threshold=' + str(best_thresh) + ' inches'
    delta_dst = (the_max_err - the_min_err)
    dst_up = delta_dst / 20
    ax.text(best_thresh + 0.15, dst_up, msg, fontsize=16, backgroundcolor='w');
    plt.savefig('My_Pseudo_Otsu_Based_On_Dr_K__Otsu_Answer.jpg', dpi=100)
    plt.show()

    return best_thresh


def create_hist_cls(title_name, csv_path, threshold, quants_per_binsize):
    # get data
    all_the_data = pd.read_csv(csv_path, index_col=False)

    # take a quick look at data
    all_the_data.head()

    the_data_values = all_the_data['Day8']

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

    plt.savefig('Histogram_Classification.jpg', dpi=100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file_path', help='Required: file path to a csv file for dataset')
    args = parser.parse_args()
    csv_path = args.csv_file_path
    threshold = otsu_classification(csv_path)
    # creating a histogram with a binsize of at least 4 * (1/32) inches
    create_hist_cls("Classification of Plants", csv_path, threshold, 4)

if __name__ == "__main__":
    main()
else:
    print("this is NOT main")

#%%
