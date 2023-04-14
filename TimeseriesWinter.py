import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
#df = pd.read_excel('TimeseriesDE.xlsx', parse_dates=['Datetime'], index_col='Datetime')
#
## Extract the data for a specific day
#day = df.loc['2019-12-11']
#
## Create the plot
#fig, ax = plt.subplots(figsize=(12, 6))
#ax.plot(day.index, day['wind_offshore_generation'])
#print(day.index)
## Customize the x-axis labels
#ticks = pd.date_range('2019-12-11', periods=25, freq='H')
#print(ticks)
#labels = [t.strftime('%H:%M') for t in ticks]
#print (labels)
#ax.set_xticks(ticks)
#ax.set_xticklabels(labels, rotation=45)
#
## Set the plot title and axis labels
#ax.set_title('Wind Offshore Generation on December 11th (DE), 2019 with capacity of 5723 MW')
#ax.set_xlabel('Time')
#ax.set_ylabel('Power Generation (MW)')
#
## Show the plot
#plt.show()
#
## Define the original time axis and demand forecast data
#
# #################################
#
## define the x-axis array with updated time axis
#
## Define the x and y arrays
#x = np.arange(0, 24)
#y = np.array([4.642,4.478,4.383,4.343,4.387,4.633,5.106,5.548,5.806,5.932,5.971,5.965,5.983,5.979,5.956,5.974,6.031,6.083,5.971,5.818,5.646,5.418,5.122,4.801])
#
## Define the new time axis
#new_time_axis = np.array(['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
#
## Interpolate the y values to the new time axis
#new_x = np.linspace(0, 23, len(new_time_axis))
#new_y = np.interp(new_x, x, y)
#print(new_y)
#
## Plot the interpolated data
#plt.plot(new_time_axis, new_y)
#plt.xlabel('Time')
#plt.ylabel('Demand Forecast (GW)')
#plt.title('NYC Demand Forecast')
#plt.show()

## Read wind offshore generation data from file
#df = pd.read_excel('TimeseriesDE.xlsx', parse_dates=['Datetime'], index_col='Datetime')
#
## Extract the data for a specific day
#day = df.loc['2019-12-11']
#
## Define the original time axis and demand forecast data
#x = np.arange(0, 24)
#y = np.array([4.642,4.478,4.383,4.343,4.387,4.633,5.106,5.548,5.806,5.932,5.971,5.965,5.983,5.979,5.956,5.974,6.031,6.083,5.971,5.818,5.646,5.418,5.122,4.801])
#
## Define the new time axis for the demand forecast data
#new_time_axis = np.array(['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
#
## Interpolate the demand forecast data to the new time axis
#new_x = np.linspace(0, 23, len(new_time_axis))
#new_y = np.interp(new_x, x, y)
#
## Create the plot
#fig, ax = plt.subplots(figsize=(12, 6))
#
## Plot the wind offshore generation data
#ax.plot(day.index, day['wind_offshore_generation'], label='Wind Offshore Generation')
#
## Plot the demand forecast data
#ax.plot(new_time_axis, new_y, label='Demand Forecast')
#
## Customize the x-axis labels
#ticks = pd.date_range('2019-12-11', periods=25, freq='H')
#labels = [t.strftime('%H:%M') for t in ticks]
#ax.set_xticks(ticks)
#ax.set_xticklabels(labels, rotation=45)
#
## Set the plot title and axis labels
#ax.set_title('Wind Offshore Generation and Demand Forecast on December 11th (DE), 2019')
#ax.set_xlabel('Time')
#ax.set_ylabel('Power Generation / Demand Forecast (MW)')
#
## Add a legend to the plot
#ax.legend()
#
## Show the plot
#plt.show()
#


# Read wind offshore generation data from file
#df = pd.read_excel('TimeseriesDE.xlsx', parse_dates=['Datetime'], index_col='Datetime')
#
## Extract the data for a specific day
#day = df.loc['2019-12-11']
#
## Define the original time axis and demand forecast data
#x = np.arange(0, 24)
#y = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])
#
## Define the new time axis for the demand forecast data
#new_time_axis = np.array(['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
#
## Convert the new time axis to a datetime object
#new_time_axis = pd.to_datetime(new_time_axis, format='%H:%M').time
#
## Get the datetime for the specific day and new time axis
#new_datetime_axis = [pd.Timestamp.combine(day.index.date[0], t) for t in new_time_axis]
#
## Interpolate the demand forecast data to the new time axis
#new_x = np.linspace(0, 23, len(new_datetime_axis))
#new_y = np.interp(new_x, x, y)
#

## Create the plot
#fig, ax = plt.subplots(figsize=(12, 6))
#
## Plot the wind offshore generation data
#ax.plot(day.index, day['wind_offshore_generation'], label='Wind Offshore Generation')
#
## Plot the demand forecast data
#ax.plot(new_datetime_axis, new_y, label='Demand Forecast')
#
## Customize the x-axis labels
#ticks = pd.date_range('2019-12-11', periods=25, freq='H')
#labels = [t.strftime('%H:%M') for t in ticks]
#ax.set_xticks(ticks)
#ax.set_xticklabels(labels, rotation=45)
#
## Set the plot title and axis labels
#ax.set_title('Wind Offshore Generation and Demand Forecast on December 11th (DE), 2019')
#ax.set_xlabel('Time')
#ax.set_ylabel('Power Generation / Demand Forecast (MW)')
#
## Add a legend to the plot
#ax.legend()
#
## Show the plot
#plt.show()

##########################

# Read wind offshore generation data from file
#df = pd.read_excel('TimeseriesDE.xlsx', parse_dates=['Datetime'], index_col='Datetime')
#
## Extract the data for a specific day
#day = df.loc['2019-12-11']
#
## Define the original time axis and demand forecast data
#x = np.arange(0, 24)
#y = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])
#
## Define the new time axis for the demand forecast data
#new_time_axis = np.array(['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
#
## Convert the new time axis to a datetime object
#new_time_axis = pd.to_datetime(new_time_axis, format='%H:%M').time
#
## Get the datetime for the specific day and new time axis
#new_datetime_axis = [pd.Timestamp.combine(day.index.date[0], t) for t in new_time_axis]
#
## Interpolate the demand forecast data to the new time axis
#new_x = np.linspace(0, 23, len(new_datetime_axis))
#new_y = np.interp(new_x, x, y)
#
## Define the time axis and power output data for wind generation
#time = day.index
#power_output = day['wind_offshore_generation']
#
## Define the time axis and power demand data for NYC
#time2 = new_datetime_axis
#NYC_demand_forecast = new_y
#
## Interpolation in order to find the area between the curves
##xx = np.linspace(min(time), max(time), 100)
#xx = np.linspace(
#    mdates.date2num(min(time)),
#    mdates.date2num(max(time)),
#    100
#)
##xx2 = np.linspace(min(time2), max(time2), 100)
#xx2 = np.linspace(
#    mdates.date2num(min(time2)),
#    mdates.date2num(max(time2)),
#    100
#)
#yy1 = np.interp(xx, time, power_output)
#yy2 = np.interp(xx2, time2, NYC_demand_forecast)
#
#area = np.trapz(np.minimum(yy1, yy2), xx) - np.trapz(np.maximum(yy1, yy2), xx)
#Area = np.trapz(yy1, xx) - np.trapz(yy2, xx)
#Power_not_available = np.trapz(yy2, xx)
#Power_in_excess = np.trapz(yy1, xx)
#
## Create the plot
#fig, ax = plt.subplots(figsize=(12, 6))
#
## Plot the wind offshore generation data
#ax.plot(time, power_output, label='Wind Offshore Generation')
#
## Plot the demand forecast data
#ax.plot(time2, NYC_demand_forecast, label='Demand Forecast')
#
## Customize the x-axis labels
#ticks = pd.date_range('2019-12-11', periods=25, freq='H')
#labels = [t.strftime('%H:%M') for t in ticks]
#ax.set_xticks(ticks)
#ax.set_xticklabels(labels, rotation=45)
#
## Set the plot title and axis labels
#ax.set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read wind offshore generation data from file
df = pd.read_excel('TimeseriesDE_15min.xlsx', parse_dates=['Datetime'], index_col='Datetime')
#start_date = '2019-01-01'
#end_date = '2019-12-31'

# create a date range from start_date to end_date
#date_range = pd.date_range(start=start_date, end=end_date, freq='D')
#for date in date_range:
    # Extract the data for a specific day
day = df.loc['2019-01-01']

# Define the original time axis and demand forecast data
x = np.arange(0, 24)
print(x)
y = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])
## time axis given in 15 minute interval
time15 = ['00:00', '00:15', '00:30', '00:45', '01:00', '01:15', '01:30', '01:45', '02:00', '02:15', '02:30', '02:45', '03:00', '03:15', '03:30', '03:45', '04:00', '04:15', '04:30', '04:45', '05:00', '05:15', '05:30', '05:45', '06:00', '06:15', '06:30', '06:45', '07:00', '07:15', '07:30', '07:45', '08:00', '08:15', '08:30', '08:45', '09:00', '09:15', '09:30', '09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45']

# Interpolate the demand forecast data to the new time axis
new_x = np.linspace(0, 24, 200)
new_y = np.interp(new_x, x, y)
print(new_x)
print(new_y)
# Define the time axis and power output data for wind generation
time = day.index
power_output = day['wind_offshore_generation'] # indes is the date and time and the values are the power output

print(time)

# Define the time axis and power demand data for NYC

NYC_demand_forecast = new_y

#print(NYC_demand_forecast)
# Convert time2 to seconds since the epoch
test=mdates.date2num(time)

# Interpolation in order to find the area between the curves
xx = np.linspace(
    mdates.date2num(min(time)),
    mdates.date2num(max(time)),
    100
)
#xx2 = np.linspace(
#    mdates.date2num(min(time2)),
#    mdates.date2num(max(time2)),
#    100
#)
yy1 = np.interp(xx, time, power_output)
#yy2 = np.interp(xx2, time2_float, NYC_demand_forecast)
print(xx)
print(xx2)
print(yy1)
print(yy2)
#area = np.trapz(np.minimum(yy1, yy2), xx) - np.trapz(np.maximum(yy1, yy2), xx)
Area = np.trapz(power_output, xx) - np.trapz(NYC_demand_forecast, xx)
Energy_from_wind = np.trapz(yy2, xx)
Energy_needed_NY = np.trapz(yy1, xx)
print(Area)
print(Energy_from_wind)
print(Energy_needed_NY)
# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the wind offshore generation data
ax.plot(time, power_output, label='Wind Offshore Generation')

# Plot the demand forecast data
ax.plot(time2, NYC_demand_forecast, label='Demand Forecast')

# Customize the x-axis labels
ticks = pd.date_range('2019-12-11', periods=25, freq='H')
labels = [t.strftime('%H:%M') for t in ticks]
ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=45)

# Set the plot title and axis labels
ax.set(title='Wind Offshore Generation vs. Demand Forecast', xlabel='Time (hours)', ylabel='Power Output (MW)')

# Add a legend to the plot
ax.legend()
#plt.plot(xx, yy1, 'r', xx, yy2, 'b')
plt.xlabel('time')
plt.ylabel('Power MW')
plt.legend(['Wind fluctuations', 'NYC energy demand', f'Excess energy over 24h= {Energy_from_wind:.2f}', f'Energy not available over 24h = {Energy_needed_NY:.2f}'])
plt.fill_between(xx, np.maximum(power_output, NYC_demand_forecast), NYC_demand_forecast, color=[0.3, 0.6, 0.3], edgecolor='none')
plt.fill_between(xx, np.minimum(power_output, NYC_demand_forecast), NYC_demand_forecast, color=[0.8, 0.8, 0.8], edgecolor='none')

plt.show()

