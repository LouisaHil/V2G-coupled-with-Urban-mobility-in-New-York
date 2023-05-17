import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from scipy.integrate import simps, quad
from datetime import datetime, timedelta

battery_capacity= 70 #kWh
charging_rate=50 #kW
charging_efficiency=0.9
Energy_consumption_miles= 34/100   #An average Tesla electric car uses around 34 kWh of electricity per 100 miles.
Efficiency = 1/Energy_consumption_miles
Smin=0.35


### compute SOC
avg_charging_rate= 50 #kW
avg_discharging_rate=-40
avg_speed_peak=7.2 #miles       between 8-10am and between 5-7pm
avg_speed=8.8



# read the csv file into a dataframe
df = pd.read_csv('new_NbofEvs_1.csv') ## depends on the month chosen
# Convert the timestamp column to pandas datetime format
df['time_interval'] = pd.to_datetime(df['time_interval'])
see=df['2013013426']
start_date = '2013-01-01'
end_date = '2013-01-03'

mask = (df['time_interval'] >= start_date) & (df['time_interval'] <= end_date)
df_filtered = df[mask]
# Filter the DataFrame for the specific month (e.g., 2013-01-01) Here you look at the exaxt range od days you would like
#mask = (df['time_interval'].dt.year == 2013) & (df['time_interval'].dt.month == 1) & (df['time_interval'].dt.day == 2)
#df_filtered = df[mask]

# Sum the availability for each medaillon_id, multiply by 15, and divide by 60
availability_sum1=df_filtered.set_index('time_interval').sum()
availability_sum = df_filtered.set_index('time_interval').sum() * 15 / 60

# Calculate not available hours
total_intervals = len(df_filtered)
notavail=(total_intervals - availability_sum1)
not_available_sum = (total_intervals - availability_sum1) * 15 / 60
# Combine available and not available hours in a new DataFrame
availability_counts = pd.concat([availability_sum, not_available_sum], axis=1).reset_index()
availability_counts.columns = ['medaillon_id', 'parking time (hours)', 'driving time (hours)']
# Compute the distance traveled given an average speed of 8.8 miles per hour
average_speed = 8.8
availability_counts['distance_traveled'] = availability_counts['driving time (hours)'] * average_speed
availability_counts['Energy_required_trips kWh']=availability_counts['distance_traveled']*Energy_consumption_miles
availability_counts = availability_counts[(availability_counts['parking time (hours)'] != 0) & (availability_counts['driving time (hours)'] != 0)]
availability_counts = availability_counts[availability_counts['medaillon_id'] != 'total']
# Calculate the average available hours and not available hours
average_available_hours = availability_counts['parking time (hours)'].mean()
average_not_available_hours = availability_counts['driving time (hours)'].mean()
average_distance_traveled = availability_counts['distance_traveled'].mean()
average_energy_required = availability_counts['Energy_required_trips kWh'].mean()
# Create a new row with the averages and append it to the DataFrame
average_row = pd.DataFrame({'medaillon_id': ['Average'],
                            'parking time (hours)': [average_available_hours],
                            'driving time (hours)': [average_not_available_hours],
                            'distance_traveled': [average_distance_traveled],
                            'Energy_required_trips kWh': [average_energy_required]})  # Set distance_traveled to None or any other value you prefer
availability_counts = availability_counts.append(average_row, ignore_index=True)

## BE careful there is many medaillon ids where we have only 0's this does not always mean that the car is always driving but that that this taxi was not operating during that day
## we only take the ones into account that are at least parked for 1 hour.
#just take away all that either have driving or parking time 0 during the whole month
print(availability_counts)