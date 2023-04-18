from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import date2num
#start_time = datetime(2019, 1, 1, 0, 0)  # Start time as a datetime object
interval = [[27.88, 46.25], [47.707, 57.21], [61.72, 74.13], [77.19, 82.03], [86.06, 87.51],
            [88.79, 97.98], [105.39, 106.20], [152.29, 168]]  # List of intervals in hours
#interval=[[27.972972972972972, 46.34534534534534], [47.7957957957958, 57.327327327327325], [61.81681681681682, 72]]

# incorporate the interval line of code into the scrapcode. this needs to be taken outta of the code from the Timeseries data of the wind
start_time = '2013-01-01 00:00:00'
#start_datetime = datetime(2013, 1, 1, 0, 0)
start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
def hours_to_date(start_datetime,interval):
    result=[]
    for iv in interval:
        start_hour = int(iv[0])
        start_minute = int((iv[0] - start_hour) * 60)
        end_hour = int(iv[1])
        end_minute = int((iv[1] - end_hour) * 60)
        start_time = start_datetime + timedelta(hours=start_hour, minutes=start_minute)
        end_time = start_datetime + timedelta(hours=end_hour, minutes=end_minute)
        result1=start_time.strftime('%Y-%m-%d %H:%M') + ' - ' + end_time.strftime('%Y-%m-%d %H:%M')
        result.append(result1)
    return result

non_availability= hours_to_date(start_datetime,interval)
print(hours_to_date(start_datetime,interval))


start_datetime = datetime(2013, 1, 1, 0, 0)  # Start of week (January 1, 2013 at 00:00)
#interval = [[27.88, 46.25], [47.707, 57.21], [61.72, 74.13], [77.19, 82.03], [86.06, 87.51],
            #[88.79, 97.98], [105.39, 106.20], [152.29, 168]]  # List of intervals in hours


# Read the data
df = pd.read_csv('new_NbofEvs_1.csv', index_col=0, parse_dates=True)

# Select specific medallion and time period
medallion = '2013000001'
start_time = '2013-01-01 00:00:00'
end_time = '2013-01-31 23:45:00'
data = df.loc[start_time:end_time, medallion]
# Identify when a driving trip starts
trip_start = (data == 0) & (data.shift(1) == 1)

# Identify when a driving trip ends
trip_end = (data == 1) & (data.shift(1) == 0)

# Get the start and end times of each driving trip
start_times = data[trip_start].index.shift(-1, freq='15min')
end_times = data[trip_end].index
##
# Check if the series starts with a 0 and add the start time to start_times if needed
if data.iloc[0] == 0:
    start_times = start_times.insert(0, data.index[0])
if data.iloc[-1] == 0 or data.iloc[-1] == 0:
    end_times = end_times.insert(len(end_times), pd.Timestamp(end_time))
# Calculate the duration of each driving trip in hours
durations = (end_times - start_times).total_seconds() / 3600
# Convert durations to a numpy array
durations_array = np.array(durations)
time_intervals = [f"{start_time1}-{end_time1}" for start_time1, end_time1 in zip(start_times, end_times)]
# Create a new DataFrame with time intervals as index and durations array as the column
new_df = pd.DataFrame({'Duration': durations_array}, index=pd.Index(time_intervals, name='Time_Interval'))

# Transform the array into a DataFrame
#new_df = pd.DataFrame(durations_array)

# Optionally, you can specify column names and index labels
#column_name = 'DurationHours'
#index_labels = ['durationHours_{}'.format(i) for i in range(durations_array.shape[0])]
#new_df.columns = [column_name]



#### pseudo code for SOC charging and discharign
## rule 1 : Only discharge when the wind is non-available otherwise always charge.
## scenario 1 : We start with an SOC at 1 and we define a minimum SOC. Minimus SOC is 0.3 .
Average_speed=10 # miles
Energy_comsumption=0.34 #kWh/miles
Efficiency=1/Energy_comsumption #miles/kWh
Energy_per_hour=Average_speed/Efficiency
new_df['Total_Energy_used']=Energy_per_hour*durations_array
SOC_min=0.35 ## we checked the longest trip takes about 27 KwH which is about 40% of the maximum battery capacity of 70 Kwh.





### plotting #####
# Create a step function plot
fig, ax = plt.subplots(figsize=(12, 6))
look=data.index
look2=data.values
ax.step(data.index, data.values, where='post')
ax.set(title='Binary Numbers for Medallion 1 2013/01/01-2013/01/08', xlabel='Time', ylabel='Status')
ax.set_ylim(-0.1, 1.1)

for interval in non_availability:
    start_str, end_str = interval.split(' - ')
    start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M')
    end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M')
    xmin = date2num(start_time)
    xmax = date2num(end_time)
    ax.axvspan(start_time, end_time, alpha=0.3, color='red')

# Show the plot
plt.show()

