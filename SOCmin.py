import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# initialoze the SOCmin at the end of the month with 0 ( meaning we have reached at the end of the month 0.
# from this compute backwards the SOC at each 15 min.
#df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col=0, parse_dates=True)


# Load the data
#df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col='time_interval')
df_cars = pd.read_csv('expanded_dataset.csv', index_col='time_interval')
df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')

# Parameters
battery_capacity = 70  # kWh
charging_rate = 50  # Assumed constant charging rate in kW -> kWh/15min = 50/4
initial_soc_min = 0  # initial SOCmin at 2013-01-31 23:45
energy_loss_per_15min_while_driving = 0.75  # This will depend on your actual data
def SOC_min(df):
    # initialize a new DataFrame to store the results
    df_soc_min = pd.DataFrame(index=df.index)

    # Process data
    for car_id in df.columns:
        # Get data for this car and reverse the order (latest first)
        df_car = df[car_id][::-1]

        # Initialize the SOCmin series with the same index as df_car
        soc_min = pd.Series(index=df_car.index)
        soc_min.iloc[0] = initial_soc_min  # set initial SOCmin

        # Iterate through each time step
        for i in range(1, len(df_car)):
            if df_car.iloc[i] == 0:  # car is driving
                soc_min.iloc[i] = soc_min.iloc[i - 1] + energy_loss_per_15min_while_driving / battery_capacity
            elif df_car.iloc[i] == 1:  # car is parked
                soc_min.iloc[i] = max(0, soc_min.iloc[i - 1] - charging_rate / 4)

        # Reverse back to original order and save to the result DataFrame
        df_soc_min[car_id] = soc_min[::-1]
    return df_soc_min
# Save to a CSV file
SOC_min_recompute=False
if SOC_min_recompute==True:
    df_soc_min=SOC_min(df_cars)
    df_soc_min.to_csv('SOCmin_data2.csv')



## 2019 ## change here for new month
start_time = '2013-01-01 00:00:00'
start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = '2013-12-31 23:45:00'
end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv('USWINDDATA_process2.csv',parse_dates=['Local_time_New_York'], delimiter=';')
df = df.sort_index()

###this is manually changing the entry. have not found another way to do it.
df.at[1639, 'Local_time_New_York'] = '2013-03-10 02:00'
df.drop_duplicates(subset='Local_time_New_York', inplace=True)
df.set_index('Local_time_New_York', inplace=True)

NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])


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
df15.index = df15.index + pd.DateOffset(years=2019 - df15.index.year.min())
# Filter out only the data for the month of January
january_data = df15[df15.index.month == 2]
print(january_data)

df_soc_min = pd.read_csv('SOCmin_data2.csv', index_col='time_interval')
print(df_soc_min)