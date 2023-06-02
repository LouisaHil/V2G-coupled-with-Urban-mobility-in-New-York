import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
# Initial parameters
num_cars = 2694
SOC = np.full(num_cars, 0.8) # Initialize all cars with SOC of 0.5
battery_capacity = 70  # Assume 50kWh battery capacity for each car
time_interval = 0.25  # 15 minutes in hours
charging_rate = 50  # kW, assume a constant rate for simplicity
SOC_min = 0.2
SOC_max = 1
driving_energy_loss=0.75
batch_size = num_cars
num_batches = 1
df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col=0, parse_dates=True)
# Assuming your existing DataFrame is called 'df'


## 2019 ## change here for new month
start_time = '2013-01-01 00:00:00'
start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = '2013-12-31 23:45:00'
end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

## 2013 ##change here for new month
#start = datetime(2013, 1, 1, 0, 0)
#end = datetime(2013, 1, 31, 23, 45)

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
january_data = df15[df15.index.month == 1]


# Initial parameters
num_cars = 2694
SOC = np.full((num_cars, len(df_cars)), 0.8)  # Initialize all cars with SOC of 0.5
battery_capacity = 50  # Assume 50kWh battery capacity for each car
time_interval = 0.25  # 15 minutes in hours
charging_rate = 10  # kW, assume a constant rate for simplicity
SOC_min = 0.2
SOC_max = 1

batch_size = num_cars
num_batches = 1

total_energy_charged = np.zeros(len(df_cars))
total_energy_discharged = np.zeros(len(df_cars))
num_cars_charged = np.zeros(len(df_cars), dtype=int)
num_cars_discharged = np.zeros(len(df_cars), dtype=int)

for time_step in range(len(df_cars)):

    # Update availability and SOC based on whether each car is driving or parked
    for car in range(num_cars):
        if df_cars.iloc[time_step, car] == 0:  # If driving
            # Assume some energy loss, update SOC
            # For simplicity, we're not considering distance driven here
            SOC[car][time_step] -= driving_energy_loss / battery_capacity
        elif df_cars.iloc[time_step, car] == 0:  # If parked
            # If car is parked, the SOC remains the same
            SOC[car][time_step] = SOC[car][time_step-1] if time_step > 0 else SOC[car][0]

    # Calculate power difference
    power_difference = january_data['Difference'][time_step]

    if power_difference > 0:  # If wind power is higher than demand
        # Charge cars, starting with the ones with the lowest SOC
        sorted_cars = np.argsort(SOC[:, time_step])
        for car in sorted_cars:
            if SOC[car][time_step] < SOC_max:
                charge_amount = min(power_difference, charging_rate * time_interval)
                SOC[car][time_step] += charge_amount / battery_capacity
                power_difference -= charge_amount
                total_energy_charged[time_step] += charge_amount
                num_cars_charged[time_step] += 1
                if power_difference <= 0:
                    break
            else:
                break

    else:  # If wind power is lower than demand
        # Discharge cars, starting with the ones with the highest SOC
        sorted_cars = np.argsort(SOC[:, time_step])[::-1]
        for car in sorted_cars:
            if SOC[car][time_step] > SOC_min:
                discharge_amount = min(-power_difference, charging_rate * time_interval)
                SOC[car][time_step] -= discharge_amount / battery_capacity
                power_difference += discharge_amount
                total_energy_discharged[time_step] += discharge_amount
                num_cars_discharged[time_step] += 1
                if power_difference >= 0:
                    break
            else:
                break

    # If demand still isn't met, add another batch of cars and continue
    if power_difference < 0:
        num_cars += batch_size
        new_soc = np.full((batch_size, len(df_cars)), 0.8)
        new_soc[:, :time_step+1] = SOC[:batch_size, :time_step+1]
        SOC = np.concatenate((SOC, new_soc))

    # If there's excess power even after charging all cars, remove a batch of cars
    elif power_difference > 0 and num_cars > batch_size:
        num_cars -= batch_size
        SOC = SOC[:num_cars]

# Convert results to pandas DataFrame
SOC_df = pd.DataFrame(SOC)
total_energy_charged_df = pd.Series(total_energy_charged)
total_energy_discharged_df = pd.Series(total_energy_discharged)
num_cars_charged_df = pd.Series(num_cars_charged)
num_cars_discharged_df = pd.Series(num_cars_discharged)