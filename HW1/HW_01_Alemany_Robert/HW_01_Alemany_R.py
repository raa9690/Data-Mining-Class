
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# filepath of the data used for this homework
DATA_FILE_PATH = "DataMining Data/Data_for_Discrete_Events (2).csv"

# Originally planned to subdivide this program into more functions, but learning
# how to do what I wanted to do took a lot longer than expected
def homeworkProgram():
    data = pd.read_csv(DATA_FILE_PATH)

    # get the numpy array of the days
    data_days = np.ceil(np.array(data['Hour']) / 24)

    # get list of the day value for each individual event
    day_events = []
    current_day = 0
    # also get an adjusted frequency table for the NumberOfEvents by day intervals (just to avoid an extra loop)
    number_of_events_by_day = []
    # with a counter for the current day
    counter = -1
    # And finally a variable to get the total frequency of the events
    total = 0
    for i in range(len(data_days)):
        # if there's been a change of day, set the new current day, and append a new element to the day frequency table
        if data_days[i] > current_day:
            current_day = data_days[i]
            number_of_events_by_day.append(0)
            counter += 1
        # for the frequency of events in a day, add it to the day_events list
        for j in range(data['NumberOfEvents'][i]):
            day_events.append(current_day)
        # also add the frequency of the current hour to the current frequency of that day
        number_of_events_on_day_i = data['NumberOfEvents'][i]
        number_of_events_by_day[counter] += number_of_events_on_day_i
        total += number_of_events_on_day_i


    # create a histogram repressenting the events in a day for the entire semester
    plt.hist(day_events, bins=70)
    plt.xlabel('Time of Event = DAY of the Semester')
    plt.ylabel('Count')
    plt.title('EVENTS = Some sort of data given to us for this homework')
    plt.savefig('histogram_of_events_by_day.png')
    # clear plot after saving it
    plt.clf()

    # Now with the frequency list of events by day, get an numpy array of the list divided by the total
    relative_frequency_of_events_by_day_array = np.array(number_of_events_by_day) / total

    # plotting original data as relative frequency line plot before Parzen Density Estimation (color blue)
    plt.plot(relative_frequency_of_events_by_day_array, linestyle='solid', color='blue', label='Relative Density of Events')

    # array to use to simulate a gaussean filter
    gaussean_filter_kernel = [ 0.06, 0.12, 0.20, 0.24, 0.20, 0.12, 0.06 ]
    # array to contain the value based on the gaussean filter
    filtered_array = [0] * len(relative_frequency_of_events_by_day_array)
    for i in range(len(relative_frequency_of_events_by_day_array)):
        for j in range(len(gaussean_filter_kernel)):
            # calculate the offset index, and only if it is a valid index, apply the gaussean filter
            offset = i + j - 3
            if (offset >= 0 and offset < len(relative_frequency_of_events_by_day_array)):
                filtered_array[i] += relative_frequency_of_events_by_day_array[offset] * gaussean_filter_kernel[j]

    # plot the Gaussean Parzen Density Estimation on top of the Relative Density plot, and save to a file
    plt.plot(filtered_array, linestyle='solid', color='red', label='Parzen Density Estimation')
    plt.xlabel('Time of Event = DAY of the Semester')
    plt.ylabel('Estimated Likelihood of Event Per Day')
    plt.title('Gaussean Parzen Density Estimation Compared to Relative Density')
    plt.legend(loc='best')
    plt.savefig('parzen_density_estimation_of_data.png')

# Thank you for making it through my mess of a program, sorry for the trouble it brought
homeworkProgram()


#%%
