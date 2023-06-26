import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np


## 2019 ## change here for new month
start_time = '2019-01-01 00:00:00'
start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = '2019-12-31 23:45:00'
end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

## 2013 ##change here for new month
start = datetime(2013, 1, 1, 0, 0)
end = datetime(2013, 1, 31, 23, 45)

df = pd.read_csv('USWINDDATA_process2.csv',parse_dates=['Local_time_New_York'], delimiter=';')
df = df.sort_index()

###this is manually changing the entry. have not found another way to do it.
df.at[1639, 'Local_time_New_York'] = '2013-03-10 02:00'
df.drop_duplicates(subset='Local_time_New_York', inplace=True)
df.set_index('Local_time_New_York', inplace=True)

NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])
df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col=0, parse_dates=True)


def interval_dataframe(df,start_time,end_time):
    #day = df.loc[start_time:end_time]
    time = df.index
    power_output=df['Electricity'].to_numpy()/1000
    return df, time, power_output
def fnum_days(start_time,end_time):
    start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    time_diff = end_datetime - start_datetime
    num_days = time_diff.days + 1 # Add one day to include January 1st
    return num_days

def NYdemand_days(num_days,NYC_demand):
    NYC_demand = np.tile(NYC_demand, num_days)
    return NYC_demand

def interpolation15(NYC_demand, power_output):
    index = pd.date_range(start=start_time, end=end_time, freq='H')
    # Create a dataframe with the hourly data and datetime index
    df = pd.DataFrame({'NYC_demand': NYC_demand, 'power_output': power_output}, index=index)
    # Resample the dataframe to 15-minute frequency and interpolate the missing values
    df_interpolated = df.resample('15T').interpolate()
    df_interpolated['Difference'] = df_interpolated['power_output'] - df_interpolated['NYC_demand']
    return df_interpolated

day, time, power_output=interval_dataframe(df, start_time, end_time)
num_days=fnum_days(start_time,end_time)
NYC_demand=NYdemand_days(num_days,NYC_demand)
df15=interpolation15(NYC_demand,power_output)
print(df15)



