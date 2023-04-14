from datetime import datetime, timedelta

#start_time = datetime(2019, 1, 1, 0, 0)  # Start time as a datetime object
interval = [[27.88, 46.25], [47.707, 57.21], [61.72, 74.13], [77.19, 82.03], [86.06, 87.51],
            [88.79, 97.98], [105.39, 106.20], [152.29, 168]]  # List of intervals in hours

start_time = '2019-01-01 00:00:00'

start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
def hours_to_date(start_time,interval):
    result=[]
    for iv in interval:
        start_hour = int(iv[0])
        start_minute = int((iv[0] - start_hour) * 60)
        end_hour = int(iv[1])
        end_minute = int((iv[1] - end_hour) * 60)

        start_time += timedelta(hours=start_hour, minutes=start_minute)
        end_time = start_time + timedelta(hours=end_hour - start_hour, minutes=end_minute - start_minute)
        result1=start_time.strftime('%Y-%m-%d %H:%M') + ' - ' + end_time.strftime('%Y-%m-%d %H:%M')
        result.append(result1)
    return result


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import date2num

start_datetime = datetime(2013, 1, 1, 0, 0)  # Start of week (January 1, 2013 at 00:00)
interval = [[27.88, 46.25], [47.707, 57.21], [61.72, 74.13], [77.19, 82.03], [86.06, 87.51],
            [88.79, 97.98], [105.39, 106.20], [152.29, 168]]  # List of intervals in hours
non_availability=[]
for iv in interval:
    start_hour = int(iv[0])
    start_minute = int((iv[0] - start_hour) * 60)
    end_hour = int(iv[1])
    end_minute = int((iv[1] - end_hour) * 60)

    # Calculate start and end datetime objects
    start_time = start_datetime + timedelta(hours=start_hour, minutes=start_minute)
    end_time = start_datetime + timedelta(hours=end_hour, minutes=end_minute)
    non_availability.append(start_time.strftime('%Y-%m-%d %H:%M') + ' - ' + end_time.strftime('%Y-%m-%d %H:%M'))
    #print(start_time.strftime('%Y-%m-%d %H:%M') + ' - ' + end_time.strftime('%Y-%m-%d %H:%M'))
# Define the non-availability intervals
print(non_availability)

# Convert the intervals to datetime objects
#non_avail_start_end = [interval.split(' - ') for interval in non_availability]
#non_avail_start_end = [(datetime.strptime(start, '%Y-%m-%d %H:%M'), datetime.strptime(end, '%Y-%m-%d %H:%M')) for start, end in non_avail_start_end]

# Read the data
df = pd.read_csv('new_NbofEvs_1.csv', index_col=0, parse_dates=True)

# Select specific medallion and time period
medallion = '2013000001'
start_time = '2013-01-01 00:00:00'
end_time = '2013-01-07 23:00:00'
data = df.loc[start_time:end_time, medallion]

# Create a step function plot
fig, ax = plt.subplots(figsize=(12, 6))
data.index
ax.step(data.index, data.values, where='post')
ax.set(title='Binary Numbers for Medallion 1 in January 2013', xlabel='Time', ylabel='Status')
ax.set_ylim(-0.1, 1.1)

for interval in non_availability:
    start_str, end_str = interval.split(' - ')
    start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M')
    end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M')
    xmin = date2num(start_time)
    xmax = date2num(end_time)
    ax.axhline(y=0.5, xmin=xmin, xmax=xmax, color='r', linestyle='--')

# Show the plot
plt.show()
# Show the plot
plt.show()




