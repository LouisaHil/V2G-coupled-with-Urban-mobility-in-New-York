import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
namedata='/Users/louisahillegaart/pythonProject1/Trip_data/trip_data_1.csv'


# def load_and_preprocess_data(namedata, month):
# #    taxi_df = pd.read_csv(namedata, parse_dates=[5, 6])
# #    taxi_df.columns = taxi_df.columns.str.strip()
# #    taxi_df = taxi_df[(taxi_df['pickup_datetime'].dt.strftime('%Y-%m') == month) & (taxi_df['dropoff_datetime'].dt.strftime('%Y-%m') == month)]
# #    taxi_df['duration'] = (taxi_df['dropoff_datetime'] - taxi_df['pickup_datetime']).dt.total_seconds() / 60
# #    taxi_df = taxi_df[['medallion', 'pickup_datetime', 'dropoff_datetime', 'duration']]
# #    return taxi_df
# #
# #def calculate_availability(df, month):
# #    unique_medallions = df['medallion'].unique()
# #    time_intervals = pd.date_range(f"{month}-01 00:00:00", f"{month}-30 23:59:00", freq='15min')
# #    availability_matrix = np.zeros((len(time_intervals) - 1, len(unique_medallions)), dtype=int)
# #
# #    for idx, medallion in enumerate(unique_medallions):
# #dallion_df = df[df['medallion'] == medallion]
#        # Create the off_intervals DataFrame
#        off_intervals = medallion_df[['dropoff_datetime']].rename(columns={'dropoff_datetime': 'start_offtime'})
#        off_intervals['end_offtime'] = medallion_df['pickup_datetime'].shift(-1)
#        off_intervals['duration'] = (off_intervals['end_offtime'] - off_intervals['start_offtime']).dt.total_seconds() / 60
#        off_intervals = off_intervals[off_intervals['duration'] > 60]
#
#        for i, row in off_intervals.iterrows():
#            overlap = (time_intervals[:-1] < row['end_offtime']) & (time_intervals[1:] > row['start_offtime'])
#            availability_matrix[overlap, idx] = 1
#            #print(availability_matrix)
#        #print(availability_matrix)
#    availability_df = pd.DataFrame(availability_matrix, columns=unique_medallions, index=time_intervals[:-1])
#    availability_df['total'] = availability_df.sum(axis=1)
#    availability_df.reset_index(inplace=True)
#    availability_df.rename(columns={'index': 'time_interval'}, inplace=True)
#    return availability_df
#
#
#month = '2013-11' #you need to change the last date in line 16 of this code depending on the month
#taxi_df = load_and_preprocess_data(namedata, month)
#availability_df = calculate_availability(taxi_df, month)
#print(availability_df.shape)
#print(availability_df[['time_interval', 'total']])
#availability_df[['time_interval', 'total']].to_csv('TotalNbofEvs_November.csv', index=False)
#availability_df.to_csv('new_NbofEvs_11.csv', index=False)

def merging(merge):
    if merge:
        # create an empty dataframe
        merged_df = pd.DataFrame()

        # loop through each file and append to the merged dataframe
        for month in range(1, 13):
            file_name = f"TotalNbofEvs_{month}.csv"  # replace with your actual file name
            df = pd.read_csv(file_name)
            merged_df = merged_df.append(df, ignore_index=True)

        # save the merged dataframe to a new CSV file
        merged_df.to_csv("merged.csv", index=False)
    else:
        return 0

statement_merge=False
statement_ev=False 
merging(statement_merge)
# Load the data from a CSV file
def plot_monthyl_ev(statement_ev):
    if statement_ev:
        data = pd.read_csv('merged.csv')

        # Convert the timestamp column to a datetime object
        data['time_interval'] = pd.to_datetime(data['time_interval'])

        # Extract the month and year from the timestamp column
        data['month'] = data['time_interval'].dt.month
        data['year'] = data['time_interval'].dt.year

        # Load the data from a CSV file
        data = pd.read_csv('merged.csv')

        # Convert the timestamp column to a datetime object
        data['time_interval'] = pd.to_datetime(data['time_interval'])

        # Extract the month and year from the timestamp column
        data['month'] = data['time_interval'].dt.month
        data['year'] = data['time_interval'].dt.year

        # Loop over each month and create a separate plot
        for year in data['year'].unique():
            for month in data['month'].unique():
                month_data = data[(data['month'] == month) & (data['year'] == year)]
                if len(month_data) > 0:
                    fig, ax = plt.subplots(figsize=(16, 10))
                    ax.plot(month_data['time_interval'], month_data['total'])
                    ax.set_xlabel('Timestamp')
                    ax.set_ylabel('Number of EVs')
                    ax.set_title(f'EV usage in {month}/{year}')
                    plt.show()
    else:
        return 0 
plot_monthyl_ev(statement_ev)
###### plotting the operating times and non operating times for one car
# Load data from CSV file

def plot_individual_avail(statement):
    if statement:
        df = pd.read_csv('new_NbofEvs_1.csv', index_col=0, parse_dates=True)

        # Select specific medallion and time period
        medallion = '2013000001'
        start_time = '2013-01-01 00:00:00'
        end_time = '2013-01-07 23:00:00'
        data = df.loc[start_time:end_time, medallion]

        # Create a step function plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(data.index, data.values, where='post')
        ax.set(title='Binary Numbers for Medallion 1 in January 2013', xlabel='Time', ylabel='Status')
        ax.set_ylim(-0.1, 1.1)

        # Show the plot
        plt.show()
    else:
        return 0
plot_individual_avail(True)


