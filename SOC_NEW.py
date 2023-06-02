import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    df_interpolated['Pdifference'] = df_interpolated['power_output'] - df_interpolated['NYC_demand']
    return df_interpolated

day, time, power_output=interval_dataframe(df, start_time, end_time)
num_days=fnum_days(start_time,end_time)
NYC_demand=NYdemand_days(num_days,NYC_demand)
df15=interpolation15(NYC_demand,power_output)
df15.index = df15.index + pd.DateOffset(years=2019 - df15.index.year.min())
df15.index = df15.index.map(lambda x: x.replace(year=2013))
# Filter out only the data for the month of January
df_15 = df15[df15.index.month == 1]
# Load the data
#df15 = pd.read_csv('df15.csv', index_col=0, parse_dates=True)
df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col='time_interval',parse_dates=True)
df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')
df_SOC_min = pd.read_csv('SOCmin_data.csv', index_col='time_interval',parse_dates=True)

battery_capacity = 70  # kWh
charging_rate = 50 / 4  # kW (charging for 15 minutes)
##this is computed with an avg speed of 9.37 miles/hour, an energyconumption of 0.32 kwh per mile
energy_loss_driving = 0.75    # energy loss per 15 minutes of driving
scaling_factor = 900

# Initialize randomly the SOC for each car
df_SOC = pd.DataFrame(
    np.random.choice(np.linspace(0, 1, 11), size=df_cars.shape),
    index=df_cars.index,
    columns=df_cars.columns
)
df_Power = pd.DataFrame(0, index=df_cars.index, columns=df_cars.columns)
df_Nb = pd.DataFrame(0, index=df_cars.index, columns=['Nb'])

# Iterate over each time step
for t in df_cars.index[1:]:
    print(t)
    parked = df_cars.loc[t] == 1
    driving = ~parked
    critical = df_SOC.shift().loc[t] <= df_SOC_min.shift().loc[t]

    # Update SOC and Power for driving cars
    df_SOC.loc[t, driving] = df_SOC.shift().loc[t, driving]-energy_loss_driving/battery_capacity
    df_Power.loc[t, driving] = 0

    # Check for Pdifference>0 or Pdifference <0
    if df15.loc[t, 'Pdifference'] > 0:  # Charging
        # Calculate SOC and Power for critical cars
        df_SOC.loc[t, critical] = df_SOC.shift().loc[t, critical] + charging_rate / battery_capacity
        df_Power.loc[t, critical] = (charging_rate * scaling_factor)/1000
        Pagg_critical = df_Power.loc[t, critical].sum()
        Pdifference_left = df15.loc[t, 'Pdifference'] - Pagg_critical

        # Sort the SOC by lowest to highest
        parked_not_critical = parked & ~critical
        SOC_sorted = df_SOC.shift().loc[t, parked_not_critical].sort_values(ascending=True)

        # Calculate new Power for all parked, not critical cars
        # power given in KW we want it in MW need to dived by 1000 to get MW
        new_Power = pd.Series((charging_rate * scaling_factor)/1000, index=SOC_sorted.index)
        cumulative_Power = new_Power.cumsum()

        # Determine the number of cars for which we need to calculate the Power
        #Nb = cumulative_Power[cumulative_Power <= Pdifference_left].count()
        Nb = cumulative_Power[cumulative_Power < abs(Pdifference_left)].count()

        # Calculate Power and SOC for the selected cars
        df_Power.loc[t, new_Power.index[:Nb]] = new_Power[:Nb]
        df_SOC.loc[t, new_Power.index[:Nb]] = np.minimum(1,df_SOC.shift().loc[t, new_Power.index[:Nb]]+ charging_rate / battery_capacity)

        # The remaining cars should have Power=0 and their SOC stays the same
        df_Power.loc[t, new_Power.index[Nb:]] = 0
        df_SOC.loc[t, new_Power.index[Nb:]]=df_SOC.shift().loc[t, new_Power.index[Nb:]]
        # Calculate the total Nb of cars charging
        df_Nb.loc[t, 'Nb'] = (df_Power.loc[t] != 0).sum() * scaling_factor

    elif df15.loc[t, 'Pdifference'] < 0:  # Discharging
        # Similar process to charging but sorted from highest to lowest SOC
        # and with a negative charging rate to represent discharging
        # Note that here we have not implemented any constraints that would
        # prevent SOC from going below 0 or above the battery capacity.

        # Calculate SOC and Power for critical cars
        df_SOC.loc[t, critical] = df_SOC.shift().loc[t, critical] + charging_rate / battery_capacity
        df_Power.loc[t, critical] = (charging_rate * scaling_factor)/1000
        Pagg_critical = df_Power.loc[t, critical].sum()
        Pdifference_left = df15.loc[t, 'Pdifference'] - Pagg_critical

        # Sort the SOC by highest to lowest
        parked_not_critical = parked & ~critical
        SOC_sorted = df_SOC.shift().loc[t, parked_not_critical].sort_values(ascending=False)

        # Calculate new Power for all parked, not critical cars
        new_Power = pd.Series((charging_rate * scaling_factor)/1000, index=SOC_sorted.index)
        cumulative_Power = new_Power.cumsum()

        # Determine the number of cars for which we need to calculate the Power
        #Nb = cumulative_Power[cumulative_Power >= Pdifference_left].count()
        Nb = cumulative_Power[cumulative_Power <= abs(Pdifference_left)].count() - 1

        # Calculate Power and SOC for the selected cars
        df_Power.loc[t, new_Power.index[:Nb]] = new_Power[:Nb]
        df_SOC.loc[t, new_Power.index[:Nb]] = np.maximum(0,df_SOC.shift().loc[t, new_Power.index[:Nb]]- charging_rate/battery_capacity)

        # The remaining cars should have Power=0 and their SOC stays the same
        df_Power.loc[t, new_Power.index[Nb:]] = 0
        df_SOC.loc[t, new_Power.index[Nb:]]=df_SOC.shift().loc[t, new_Power.index[Nb:]]

        # Calculate the total Nb of cars discharging
        df_Nb.loc[t, 'Nb'] = ((df_Power.loc[t] != 0) & ~critical).sum() * scaling_factor

# Save the final results
df_SOC.to_csv('df_SOC_try.csv')
df_Power.to_csv('df_Power_try.csv')
df_Nb.to_csv('df_Nb_try.csv')