import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse;

# not much to add here, basic argparse doing argparse things
parser = argparse.ArgumentParser()
parser.add_argument('csv_file_path', help='Required: file path to a csv file for dataset')
parser.add_argument('-quant', help='Required: tag marking next value as the quantization value',
                    required=True, type=float)
parser.add_argument('-verbose', help='increase output verbosity', action='store_true')
# added an extra optional argument to save the fig, the value is the path of the savedFig (i.e. test_file/test1.png)
parser.add_argument('-savefig', help='tag marking next value as where to save the fig')
args = parser.parse_args()


# function to create the histogram using the parser arguments saved in args
def create_hist():
    csv_file_path = args.csv_file_path
    binSize = args.quant
    data = pd.read_csv(csv_file_path)
    # calculate the number of bins needed to go from the smallest value inclusive, to the largest value rounded up
    binCount = np.ceil((data['D8'].max() - data['D8'].min()) / float(binSize))
    if args.verbose:
        # print all the data in the dataframe
        print("\tdata:")
        print(data)
        # print all the data in day 8 in the dataframe
        print("\tdata Day 8:")
        print(data['D8'])
        # print the quantization value used (added this here just as a final confirmation that the proper value is
        # being used)
        print("\tquantization:")
        print(binSize)
        # prints the min & max values of day 8 from the data frame
        print("\tmin & max values day 8:")
        print(data['D8'].min())
        print(data['D8'].max())
        # print the bin count of the final histogram using the quant value (first bin is [minValue, minValue+quant) )
        print("\tbin count:")
        print(binCount)
    # using pandas hist function means that the first bin starts at the smallest value,
    # can be changed to start at 0, but I prefer the values to start at the smallest one
    data.hist(column='D8', bins=int(binCount))
    plt.xlabel('Plant Length (inches)')
    plt.ylabel('Counts')
    plt.title("HW_02: Histogram of Plant Height on Day 8 of the Professor's Lawn")
    if args.savefig is not None:
        copy_plt = plt.gcf()
        copy_plt.savefig(args.savefig)
    plt.show()


def run():
    create_hist()
    return 0


run()
