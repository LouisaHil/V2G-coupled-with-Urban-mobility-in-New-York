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
df = df.drop(df.tail(5).index)

###this is manually changing the entry. have not found another way to do it.
df.at[1639, 'Local_time_New_York'] = '2013-03-10 02:00'
#df.at[8760, 'Local_time_New_York'] = '2013-12-31 19:00'
#df.at[8761, 'Local_time_New_York'] = '2013-12-31 20:00'
#df.at[8762, 'Local_time_New_York'] = '2013-12-31 21:00'
#df.at[8763, 'Local_time_New_York'] = '2013-12-31 22:00'
#df.at[8764, 'Local_time_New_York'] = '2013-12-31 23:00'

df.drop_duplicates(subset='Local_time_New_York', inplace=True)
df.set_index('Local_time_New_York', inplace=True)

#NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])
###################
df_NY = pd.read_csv('Actual_Load_NY.csv', parse_dates=['RTD End Time Stamp'], delimiter=';')
# Truncate the seconds and convert the column to pandas datetime format
df_NY['RTD End Time Stamp'] = pd.to_datetime(df_NY['RTD End Time Stamp']).dt.floor('15T')
#df.drop_duplicates(subset=['RTD End Time Stamp'], keep='first')
# Set 'RTD End Time Stamp' as the index
df_NY.set_index('RTD End Time Stamp', inplace=True)
# Drop duplicates from the index
df_NY = df_NY[~df_NY.index.duplicated(keep='first')]
# Generate the expected datetime range with a frequency of 15 minutes
expected_range = pd.date_range(start=df_NY.index.min(), end=df_NY.index.max(), freq='15T')
# Reindex with the expected datetime range
df_NY = df_NY.reindex(expected_range)
# Interpolate missing values
df_NY.interpolate(method='pad', inplace=True)
df_NY = df_NY.drop(df_NY.tail(1).index)
df_NY = df_NY.drop(['Zone Name', 'Zone PTID'], axis=1)

###################

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

def interpolation15(power_output):
    index = pd.date_range(start=start_time, end=end_time, freq='H')
    # Create a dataframe with the hourly data and datetime index
    df = pd.DataFrame({'power_output': power_output}, index=index)
    # Resample the dataframe to 15-minute frequency and interpolate the missing values
    df_interpolated = df.resample('15T').interpolate()
    return df_interpolated

day, time, power_output=interval_dataframe(df, start_time, end_time)
num_days=fnum_days(start_time,end_time)
#NYC_demand=NYdemand_days(num_days,NYC_demand)

df15=interpolation15(power_output)
new_index = pd.DatetimeIndex([date.replace(year=2013) for date in df15.index])
df15.index = new_index
#df15.index = df15.index + pd.DateOffset(years=2019 - df15.index.year.min())

##### we need to add three rows for the time 23:15, 23:30, 23:45 of the last day of the year
new_rows = pd.DataFrame([df15.iloc[-1]] * 3, columns=df15.columns)
new_rows.index = pd.date_range(df15.index[-1], periods=4, freq='15T')[1:]
df15 = df15.append(new_rows)
df15['NYC_demand'] = df_NY['RTD Actual Load'].values
df15['Pdifference'] = df15['power_output']-df15['NYC_demand']
##### we need to add three rows for the time 23:15, 23:30, 23:45 of the last day of the year
#new_rows = pd.DataFrame([df15.iloc[-1]] * 3, columns=df15.columns)
#new_rows.index = pd.date_range(df15.index[-1], periods=4, freq='15T')[1:]
#df15 = df15.append(new_rows)
# Filter out only the data for the month
month=1
df_15 = df15[df15.index.month == month]
# Load the data
#df15 = pd.read_csv('df15.csv', index_col=0, parse_dates=True)


#df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col='time_interval',parse_dates=True)
df_cars = pd.read_csv('20SyntheticJanuaryData.csv', index_col='time_interval',parse_dates=True)
print('done loading trip data')
#df_cars = df_cars.drop('total',axis=1)
df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')
df_cars = df_cars[df_cars.index.month == month]
## year
df_SOC_min = pd.read_csv('BIGSOCmin_data.csv', index_col='time_interval',parse_dates=True)
df_SOC_min = df_SOC_min[df_SOC_min.index.month == month]
print('done loading socmin')
##month
#df_SOC_min = pd.read_csv('SOCmin_data.csv', index_col='time_interval',parse_dates=True)

#df_SOC_min = df_SOC_min.drop('total',axis=1)
battery_capacity = 70  # kWh
charging_rate = 50 / 4  # kW (charging for 15 minutes)
##this is computed with an avg speed of 9.37 miles/hour, an energyconumption of 0.32 kwh per mile
discharging_rate = 50 / 4
energy_loss_driving = 0.75    # energy loss per 15 minutes of driving
scaling_factor = 17
sample_size=2694
#Max_Nb_cars=331*2694

# Initialize randomly the SOC for each car
df_SOC = pd.DataFrame(
    np.random.choice(np.linspace(0.1, 1, 10), size=df_cars.shape),
    index=df_cars.index,
    columns=df_cars.columns
)
df_Power = pd.DataFrame(0, index=df_cars.index, columns=df_cars.columns)
df_Nb_discharge = pd.DataFrame(0, index=df_cars.index, columns=['Nb'])
df_Nb_charge = pd.DataFrame(0, index=df_cars.index, columns=['Nb'])
df_CarCounts = pd.DataFrame(0, index=df_cars.index, columns=['Charging', 'Discharging', 'Critical','Driving','Parked'])
uncertainty=0.01
# Iterate over each time step
for t in df_cars.index[1:]:
    print(t)
    parked = df_cars.loc[t] == 1
    driving = ~parked
    critical = (df_SOC.shift().loc[t] <= df_SOC_min.shift().loc[t]) & (df_cars.loc[t]==0)
    #parked = parked.drop('total')
    #driving = driving.drop('total')
    #critical = critical.drop('total')
    # Update SOC and Power for driving cars
    df_SOC.loc[t, driving] = df_SOC.shift().loc[t, driving]-energy_loss_driving/battery_capacity
    df_Power.loc[t, driving] = 0

    # Check for Pdifference>0 or Pdifference <0
    if df15.loc[t, 'Pdifference'] > 0:  # Charging
        # Calculate SOC and Power for critical cars
        df_SOC.loc[t, critical] = np.minimum(1,df_SOC.shift().loc[t, critical] + charging_rate / battery_capacity)
        df_Power.loc[t, critical] = -(charging_rate * scaling_factor)/1000
        Pagg_critical = df_Power.loc[t, critical].sum()
        #print('p aggregated critical',Pagg_critical)
        Pdifference_left = df15.loc[t, 'Pdifference'] + Pagg_critical
        #print('charging differnce left ',Pdifference_left)
        if Pdifference_left < 0:  # Need to discharge some cars because of excess power
            # Sort the SOC by highest to lowest
            available_discharge=(df_SOC.shift().loc[t]!=0 & ~critical)
            condition = df_SOC.shift().loc[t, parked] > discharging_rate / battery_capacity
            SOC_sorted = df_SOC.shift().loc[t, parked & available_discharge & condition].sort_values(ascending=False)

            # Calculate new Power for all parked, not critical cars
            # power given in KW we want it in MW need to dived by 1000 to get MW
            new_Power = pd.Series((discharging_rate * scaling_factor) / 1000, index=SOC_sorted.index)
            cumulative_Power = new_Power.cumsum()

            # Determine the number of cars for which we need to calculate the Power
            Nb = cumulative_Power[cumulative_Power <= abs(Pdifference_left)].count() +1

            # Update SOC and Power for the cars that will be discharging
            df_SOC.loc[t, new_Power.index[:Nb]] = df_SOC.shift().loc[t, new_Power.index[:Nb]] - discharging_rate / battery_capacity
            df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]
            # The remaining cars should have Power=0 and their SOC stays the same
            # Convert your mask into pandas Series with car_id as index if they are not
            # This is necessary to match the index for the boolean indexing to work
            #mask1 = driving & critical
            #not_discharging = new_Power.index[Nb:]
            #filtered_car_ids = not_discharging[~mask1[not_discharging]]
            # Update df_SOC for car_ids in 'result'
            df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]
            # Update df_Power for car_ids in 'result'
            df_Power.loc[t, new_Power.index[Nb:]] = 0
        else:
            # Normal charging operations continue here
            # Sort the SOC by lowest to highest

            parked_not_critical = parked & ~critical
            SOC_sorted = df_SOC.shift().loc[t, parked_not_critical ].sort_values(ascending=True)

            # Calculate new Power for all parked, not critical cars
            # power given in KW we want it in MW need to dived by 1000 to get MW
            new_Power = pd.Series((-charging_rate * scaling_factor) / 1000, index=SOC_sorted.index)
            cumulative_Power = new_Power.cumsum()


            # Determine the number of cars for which we need to calculate the Power
            Nb = cumulative_Power[-cumulative_Power < abs(Pdifference_left)].count()

            # Update SOC and Power for the cars that will be charging
            df_SOC.loc[t, new_Power.index[:Nb]] = np.minimum(1, df_SOC.shift().loc[t, new_Power.index[:Nb]] + charging_rate / battery_capacity)
            df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]

            df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]
            # Update df_Power for car_ids in 'result'
            df_Power.loc[t, new_Power.index[Nb:]] = 0
            # The remaining cars should have Power=0 and their SOC stays the same
            #charging_mask = pd.Series(df_Power.index.isin(new_Power.index[:Nb]), index=df_Power.index)
            #not_charging_nor_critical = ~(charging_mask | critical.loc[new_Power.index] | driving.loc[new_Power.index])
            #df_SOC.loc[t, not_charging_nor_critical] = df_SOC.shift().loc[t, not_charging_nor_critical]
            #df_Power.loc[t, not_charging_nor_critical] = 0
            #df_Nb_charge.loc[t, 'Nb'] = -(df_Power.loc[t] < 0).sum() * scaling_factor


        df_CarCounts.loc[t, 'Charging'] = ((df_Power.loc[t] < 0) & ~critical & parked).sum() * scaling_factor # Negative power means charging
        df_CarCounts.loc[t, 'Discharging'] = (df_Power.loc[t] > 0).sum() * scaling_factor # Positive power means discharging## Sort the SOC by lowest to highest
        df_CarCounts.loc[t, 'Critical'] = critical.sum() * scaling_factor # Already defined critical as a boolean Series, so sum gives count of True values#parked_not_critical = parked & ~critical
        df_CarCounts.loc[t, 'Driving'] = driving.sum()
        df_CarCounts.loc[t, 'Parked_neutral'] = parked.sum() - (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] + df_CarCounts.loc[t, 'Critical'])/scaling_factor
        df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging']+df_CarCounts.loc[t, 'Discharging']+df_CarCounts.loc[t, 'Critical'])/(1-df_CarCounts.loc[t, 'Driving']/sample_size)
        #
    elif df15.loc[t, 'Pdifference'] < 0:  # Discharging
        # Similar process to charging but sorted from highest to lowest SOC
        # and with a negative charging rate to represent discharging
        # Note that here we have not implemented any constraints that would
        # prevent SOC from going below 0 or above the battery capacity.
        # Calculate SOC and Power for critical cars
        df_SOC.loc[t, critical] = np.minimum(1,df_SOC.shift().loc[t, critical] + charging_rate / battery_capacity)
        df_Power.loc[t, critical] = -(charging_rate * scaling_factor)/1000
        Pagg_critical = df_Power.loc[t, critical].sum()
        Pdifference_left = df15.loc[t, 'Pdifference'] + Pagg_critical
        #print('p aggregated critical', Pagg_critical)
        #print('charging differnce left ', Pdifference_left)
        # Sort the SOC by highest to lowest
        available_discharge = (df_SOC.shift().loc[t] != 0 & ~critical)
        condition= df_SOC.shift().loc[t, parked]> discharging_rate/battery_capacity
        parked_not_critical = parked & ~critical
        SOC_sorted = df_SOC.shift().loc[t, parked & available_discharge & condition].sort_values(ascending=False)

        # Calculate new Power for all parked, not critical cars
        new_Power = pd.Series((discharging_rate * scaling_factor)/1000, index=SOC_sorted.index)
        cumulative_Power = new_Power.cumsum()

        # Determine the number of cars for which we need to calculate the Power
        #Nb = cumulative_Power[cumulative_Power >= Pdifference_left].count()
        Nb = cumulative_Power[cumulative_Power <= abs(Pdifference_left)].count() +1   ### actually be carefeful because P_difference left can also be negativ when there are too many cars critical and then taking the critical value makes it worse!!

        # Calculate Power and SOC for the selected cars
        df_Power.loc[t, new_Power.index[:Nb]] = new_Power[:Nb]
        df_SOC.loc[t, new_Power.index[:Nb]] = df_SOC.shift().loc[t, new_Power.index[:Nb]]- discharging_rate/battery_capacity          ## this should not be correct

        #mask1 = driving & critical
        # critical_series = pd.Series(mask1, index=new_Power.index)
        # driving_series = pd.Series(driving, index=new_Power.index)
        # Get the car_ids that are not discharging
        #not_discharging = new_Power.index[Nb:]
        #filtered_car_ids = not_discharging[~mask1[not_discharging]]
        # Update df_SOC for car_ids in 'result'
        df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]        # Update df_Power for car_ids in 'result'
        df_Power.loc[t, new_Power.index[Nb:]] = 0

        # Calculate the total Nb of cars discharging
        #df_Nb_discharge.loc[t, 'Nb'] = ((df_Power.loc[t] > 0) & ~critical).sum() * scaling_factor
        df_CarCounts.loc[t, 'Charging'] = ((df_Power.loc[t] < 0) & ~critical & parked).sum() * scaling_factor # Negative power means charging
        df_CarCounts.loc[t, 'Discharging'] = (df_Power.loc[t] > 0).sum() * scaling_factor # Positive power means discharging## Sort the SOC by lowest to highest
        df_CarCounts.loc[t, 'Critical'] = critical.sum() * scaling_factor # Already defined critical as a boolean Series, so sum gives count of True values#parked_not_critical = parked & ~critical
        df_CarCounts.loc[t, 'Driving'] = driving.sum()
        df_CarCounts.loc[t, 'Parked_neutral'] = parked.sum()- (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] + df_CarCounts.loc[t, 'Critical'])/scaling_factor
        df_CarCounts.loc[t, 'Notavailbutparked'] = (~condition).sum()
        df_CarCounts.loc[t, 'margin'] = df_CarCounts.loc[t, 'Parked_neutral']-df_CarCounts.loc[t, 'Notavailbutparked']
        df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging']+df_CarCounts.loc[t, 'Discharging']+df_CarCounts.loc[t, 'Critical'])/(1-df_CarCounts.loc[t, 'Driving']/sample_size-df_CarCounts.loc[t, 'Notavailbutparked']/sample_size)


# Save the final results

df_Power_sum = df_Power.sum(axis=1).to_frame().rename(columns={0: 'Sum'})
joined_df = df15.join(df_Power_sum)
joined_df=joined_df.join(df_CarCounts)
#joined_df=joined_df.join(df_Nb_charge)
#df_Power.to_csv('df_Power_month.csv')
df_SOC.to_csv('df_BIGSOC_month1.csv')
joined_df.to_csv('df_BIGPower_month1csv')
#df_Nb.to_csv('df_Nb3.csv')