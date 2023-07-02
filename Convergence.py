import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

recompute_dfNY_WIND=False

compute_convergence=True
plot_true=False

#month=5
battery_capacity = 70  # kWh
charging_rate = 50 / 4  # kW (charging for 15 minutes)
discharging_rate = 50 / 4
energy_loss_driving = 0.75    # energy loss per 15 minutes of driving

#min_scaling_factor = 300
sample_size=2694

#raw wind for 9000MW and NY demand
df_wind = pd.read_csv('USWINDDATA_process2.csv',parse_dates=['Local_time_New_York'], delimiter=';')
df_NY = pd.read_csv('Actual_Load_NY.csv', parse_dates=['RTD End Time Stamp'], delimiter=';')


## datset car availability
df_cars = pd.read_csv('expanded_dataset.csv', index_col='time_interval')

#df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col='time_interval',parse_dates=True)

df15 = pd.read_csv('Wind_NY_Power', index_col=0, parse_dates=True)
#df_15 = df15[df15.index.month == month]

#df_SOC_min = pd.read_csv('SOCmin_data2.csv'  , index_col='time_interval',parse_dates=True)
#df_SOC_min = df_SOC_min.drop('total',axis=1)

df_SOC_min = pd.read_csv('2_scaleSOCmin_data.csv', index_col='time_interval',parse_dates=True)
#df_SOC_min = df_SOC_min[df_SOC_min.index.month == month]


def load_and_extract_month(df_cars,month):
    df_cars.index = pd.to_datetime(df_cars.index)
    df_cars = df_cars.sort_values(by='time_interval')
    df_cars = df_cars[df_cars.index.month == month]
    df_cars = df_cars.drop('total', axis=1)
    return df_cars

#if load_extract_month:
    #df_cars=load_and_extract_month(df_cars,month)
def dfNY_WIND(df,df_NY):
    start_time = '2013-01-01 00:00:00'
    start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = '2013-12-31 23:45:00'
    end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    #df = pd.read_csv('USWINDDATA_process2.csv',parse_dates=['Local_time_New_York'], delimiter=';')
    df = df.sort_index()
    df = df.drop(df.tail(5).index)

    ###this is manually changing the entry. have not found another way to do it.
    df.at[1639, 'Local_time_New_York'] = '2013-03-10 02:00'

    df.drop_duplicates(subset='Local_time_New_York', inplace=True)
    df.set_index('Local_time_New_York', inplace=True)

    #NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])
    ###################
    #df_NY = pd.read_csv('Actual_Load_NY.csv', parse_dates=['RTD End Time Stamp'], delimiter=';')
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

    df15.to_csv('Wind_NY_Power')
# Filter out only the data for the month
if recompute_dfNY_WIND:
    dfNY_WIND(df_wind,df_NY)




def critical_SOC_power_calculator(t,scaling_factor, critical, rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts):
    df_SOC.loc[t, critical] = df_SOC_shifted.loc[t, critical] + rate/battery_capacity
    df_Power.loc[t, critical] = -(charging_rate * scaling_factor)/1000
    #df_CarCounts[t, 'noscale_critical']=critical.sum()
    df_CarCounts.loc[t, 'noscale_critical'] = critical.sum()

def charge(t, scaling_factor,parked, critical, Pdifference_left, rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts):
    parked_not_critical = parked & ~critical
    SOC_sorted = df_SOC.shift().loc[t, parked_not_critical].sort_values(ascending=True)
    new_Power = pd.Series((-rate * scaling_factor) / 1000, index=SOC_sorted.index)
    cumulative_Power = new_Power.cumsum()
    # Determine the number of cars for which we need to calculate the Power
    Nb = cumulative_Power[-cumulative_Power < abs(Pdifference_left)].count()
    #df_CarCounts[t, 'noscale_charging']=Nb
    df_CarCounts.loc[t, 'noscale_charging'] = Nb
    # Update SOC and Power for the cars that will be charging
    df_SOC.loc[t, new_Power.index[:Nb]] = np.minimum(1, df_SOC_shifted.loc[t, new_Power.index[:Nb]] + rate / battery_capacity)
    df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]
    df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    df_Power.loc[t, new_Power.index[Nb:]] = 0

def discharge(t, scaling_factor,condition, parked, critical, Pdifference_left, rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts):
    # Sort the SOC by highest to lowest
    # Sort the SOC by highest to lowest
    parked_not_critical = parked & ~critical

    available_discharge = (df_SOC_shifted.loc[t] != 0 & ~critical)
    SOC_sorted = df_SOC_shifted.loc[t, parked & available_discharge & condition].sort_values(ascending=False)


    # Calculate new Power for all parked, not critical cars
    # power given in KW we want it in MW need to dived by 1000 to get MW
    new_Power = pd.Series((rate * scaling_factor) / 1000, index=SOC_sorted.index)
    cumulative_Power = new_Power.cumsum()

    # Determine the number of cars for which we need to calculate the Power
    Nb = cumulative_Power[cumulative_Power <= abs(Pdifference_left)].count() + 1
    #df_CarCounts[t, 'noscale_discharging']=Nb
    df_CarCounts.loc[t, 'noscale_discharging'] = Nb
    # Update SOC and Power for the cars that will be discharging
    df_SOC.loc[t, new_Power.index[:Nb]] = df_SOC_shifted.loc[t, new_Power.index[:Nb]] - rate / battery_capacity
    df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]

    df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC_shifted.loc[t, new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    df_Power.loc[t, new_Power.index[Nb:]] = 0


def charge_update_car_counts(t, scaling_factor,parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size):
    df_CarCounts.loc[t, 'Charging'] = ((df_Power.loc[
                                            t] < 0) & ~critical & parked).sum() * scaling_factor  # Negative power means charging
    df_CarCounts.loc[t, 'Discharging'] = (df_Power.loc[
                                              t] > 0).sum() * scaling_factor  # Positive power means discharging## Sort the SOC by lowest to highest
    df_CarCounts.loc[
        t, 'Critical'] = critical.sum() * scaling_factor  # Already defined critical as a boolean Series, so sum gives count of True values#parked_not_critical = parked & ~critical
    df_CarCounts.loc[t, 'Driving'] = driving.sum()
    df_CarCounts.loc[t, 'Parked_neutral'] = parked.sum() - (
                df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] + df_CarCounts.loc[
            t, 'Critical']) / scaling_factor
    df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] +
                                           df_CarCounts.loc[t, 'Critical']) / (
                                                      1 - df_CarCounts.loc[t, 'Driving'] / sample_size)
    df_CarCounts.loc[t, 'Notavailbutparked']=0
    df_CarCounts.loc[t, 'margin']=1

    #

def discharge_update_car_counts(t,scaling_factor,condition, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size):
    df_CarCounts.loc[t, 'Charging'] = ((df_Power.loc[
                                            t] < 0) & ~critical & parked).sum() * scaling_factor  # Negative power means charging
    df_CarCounts.loc[t, 'Discharging'] = (df_Power.loc[
                                              t] > 0).sum() * scaling_factor  # Positive power means discharging## Sort the SOC by lowest to highest
    df_CarCounts.loc[
        t, 'Critical'] = critical.sum() * scaling_factor  # Already defined critical as a boolean Series, so sum gives count of True values#parked_not_critical = parked & ~critical
    df_CarCounts.loc[t, 'Driving'] = driving.sum()
    df_CarCounts.loc[t, 'Parked_neutral'] = parked.sum() - (
                df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] + df_CarCounts.loc[
            t, 'Critical']) / scaling_factor
    df_CarCounts.loc[t, 'Notavailbutparked'] = (~condition).sum()
    df_CarCounts.loc[t, 'margin'] = df_CarCounts.loc[t, 'Parked_neutral'] - df_CarCounts.loc[t, 'Notavailbutparked']
    df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] +
                                           df_CarCounts.loc[t, 'Critical']) / (
                                                      1 - df_CarCounts.loc[t, 'Driving'] / sample_size -
                                                      df_CarCounts.loc[t, 'Notavailbutparked'] / sample_size)


def smart_charging(scaling_factor,month,df15,df_SOC_min,df_cars):
    print(scaling_factor)
    df15 = df15[df15.index.month == month]
    df_SOC_min = df_SOC_min[df_SOC_min.index.month == month]
    df_SOC_min_shifted = df_SOC_min.shift()
    df_cars = load_and_extract_month(df_cars,month)

    # Initialize randomly the SOC for each car
    df_SOC = pd.DataFrame(
        np.random.choice(np.linspace(0.1, 1, 10), size=df_cars.shape),
        index=df_cars.index,
        columns=df_cars.columns
    )

    df_Power = pd.DataFrame(0, index=df_cars.index, columns=df_cars.columns)
    df_CarCounts = pd.DataFrame(0, index=df_cars.index,
                                columns=['Charging', 'Discharging', 'Critical', 'Driving', 'Parked',
                                         'noscale_discharging', 'noscale_charging', 'noscale_critical'])
    df_SOC_shifted = df_SOC.shift()
    #################################
    # Iterate over each time step
    for t in df_cars.index[1:]:
        #print(t)
        parked = df_cars.loc[t] == 1
        driving = ~parked
        critical = (df_SOC_shifted.loc[t] <= df_SOC_min_shifted.loc[t]) & (df_cars.loc[t]==0)

        # Update SOC and Power for driving cars
        df_SOC.loc[t, driving] = df_SOC_shifted.loc[t, driving] - energy_loss_driving/battery_capacity
        df_Power.loc[t, driving] = 0

        # Check for Pdifference>0 or Pdifference <0
        if df15.loc[t, 'Pdifference'] > 0:  # Charging
            # Calculate SOC and Power for critical cars
            critical_SOC_power_calculator(t,scaling_factor, critical, charging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts)
            Pagg_critical = df_Power.loc[t, critical].sum()
            Pdifference_left = df15.loc[t, 'Pdifference'] + Pagg_critical
            # If there's remaining power left after charging critical cars, proceed with parked cars
            if Pdifference_left > 0:
                charge(t, scaling_factor,parked, critical, Pdifference_left, charging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts)
            if Pdifference_left < 0:
                condition = df_SOC_shifted.loc[t, parked] > discharging_rate / battery_capacity
                discharge(t,scaling_factor,condition, parked, critical, Pdifference_left, discharging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts)
            charge_update_car_counts(t,scaling_factor, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size)

        elif df15.loc[t, 'Pdifference'] < 0:  # Discharging
            # Calculate SOC and Power for critical cars
            critical_SOC_power_calculator(t,scaling_factor, critical, charging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts)
            Pagg_critical = df_Power.loc[t, critical].sum()
            Pdifference_left = df15.loc[t, 'Pdifference'] + Pagg_critical
            condition = df_SOC_shifted.loc[t, parked] > discharging_rate / battery_capacity

            discharge(t,scaling_factor,condition, parked, critical, Pdifference_left, discharging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power,df_CarCounts)

            discharge_update_car_counts(t,scaling_factor,condition, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size)
    #convergence = convergence.append(pd.Series([scaling_factor,df_CarCounts['Total_needed'].max()]), ignore_index=True)
    #df_Power_sum = df_Power.sum(axis=1).to_frame().rename(columns={0: 'Sum'})
    #joined_df = df15.join(df_Power_sum)
    #joined_df=joined_df.join(df_CarCounts)

    result=math.ceil(df_CarCounts['Total_needed'].max())
    median=math.ceil(df_CarCounts['Total_needed'].median())
    margin=df_CarCounts['margin'].min()
    # Find the index at which 'total' is maximized
    max_total_index = df_CarCounts['Total_needed'].idxmax()
    # Use this index to get the corresponding value in the 'charging' columnreturn result,median,margin, df_Nb_cars_noscale
    Nb = df_CarCounts.loc[max_total_index, 'noscale_discharging'] +  df_CarCounts.loc[max_total_index, 'noscale_charging']+ df_CarCounts.loc[max_total_index, 'noscale_critical']

    p1=round(df_CarCounts.loc[max_total_index, 'Driving']/sample_size,3)
    p2=round(df_CarCounts.loc[max_total_index, 'Notavailbutparked']/sample_size,3)
    Nb4=math.ceil(Nb/(1-p1-p2))
    Nb2= math.ceil(df_CarCounts.loc[max_total_index, 'Driving']/sample_size*Nb4)
    Nb3=math.ceil(df_CarCounts.loc[max_total_index, 'Notavailbutparked']/sample_size*Nb4)
    return result, median, margin, Nb, Nb2, Nb3,Nb4, p1, p2


    #print(convergence)

def fcompute_convergence(min_scaling_factor,month,df15,df_SOC_min,df_cars):
    if compute_convergence:
        convergence = pd.DataFrame(columns=['scaling_factor', 'Total_needed_max', 'Total_needed_median','Energy_coverage'])
        for scaling_factor in range(50000, min_scaling_factor+19700, -20000):
            result, median, margin, Nb, Nb2, Nb3,Nb4, p1, p2  = smart_charging(scaling_factor,month,df15,df_SOC_min,df_cars)
            convergence = convergence.append(
                {'scaling_factor': scaling_factor, 'Total_needed_max': result, 'Total_needed_median': median,'Energy_coverage':margin,'Nb_indivual_cars':Nb, 'Nb_driving':Nb2,'Nb_parked':Nb3, 'prob_driving': p1, 'prob_parking':p2,'DatasetNbcars':Nb4},
                ignore_index=True)
        for scaling_factor in range(min_scaling_factor+19700, min_scaling_factor+1700, -2000):
            result, median, margin, Nb, Nb2, Nb3,Nb4, p1, p2  = smart_charging(scaling_factor,month,df15,df_SOC_min,df_cars)
            convergence = convergence.append(
                {'scaling_factor': scaling_factor, 'Total_needed_max': result, 'Total_needed_median': median,'Energy_coverage':margin,'Nb_indivual_cars':Nb, 'Nb_driving':Nb2,'Nb_parked':Nb3, 'prob_driving': p1, 'prob_parking':p2,'DatasetNbcars':Nb4},
                ignore_index=True)

        for scaling_factor in range(min_scaling_factor+1700, min_scaling_factor+400, -250):
            result, median, margin, Nb, Nb2, Nb3,Nb4, p1, p2  = smart_charging(scaling_factor,month,df15,df_SOC_min,df_cars)
            convergence = convergence.append(
                {'scaling_factor': scaling_factor, 'Total_needed_max': result, 'Total_needed_median': median,'Energy_coverage':margin,'Nb_indivual_cars':Nb, 'Nb_driving':Nb2,'Nb_parked':Nb3, 'prob_driving': p1, 'prob_parking':p2,'DatasetNbcars':Nb4},
                ignore_index=True)
        for scaling_factor in range(min_scaling_factor+400, min_scaling_factor+50, -50):
            result, median, margin, Nb, Nb2, Nb3,Nb4, p1, p2  = smart_charging(scaling_factor,month,df15,df_SOC_min,df_cars)
            convergence = convergence.append(
                {'scaling_factor': scaling_factor, 'Total_needed_max': result, 'Total_needed_median': median,'Energy_coverage':margin,'Nb_indivual_cars':Nb, 'Nb_driving':Nb2,'Nb_parked':Nb3, 'prob_driving': p1, 'prob_parking':p2,'DatasetNbcars':Nb4},
                ignore_index=True)
        for scaling_factor in range(min_scaling_factor+50, min_scaling_factor-10, -5):
            result, median, margin, Nb, Nb2, Nb3,Nb4, p1, p2  = smart_charging(scaling_factor,month,df15,df_SOC_min,df_cars)
            convergence = convergence.append(
                {'scaling_factor': scaling_factor, 'Total_needed_max': result, 'Total_needed_median': median,'Energy_coverage':margin,'Nb_indivual_cars':Nb, 'Nb_driving':Nb2,'Nb_parked':Nb3, 'prob_driving': p1, 'prob_parking':p2,'DatasetNbcars':Nb4},
                ignore_index=True)
        filename = 'convergence{}.csv'.format(month)
        convergence.to_csv(filename, index=False)


# dictionary mapping each month to its minimum scaling factor
# the min_scaling factors are constraints given by the Enegrgy deficiancy/surplus
min_scaling_factors = {
    1:260,
    2:245,
    3:245,
    4:200,
    5:320,
    6:320,
    7:360,
    8:380,
    9:335,
    10:330,
    11:330,
    12:280
}



#if compute_convergence:
#    fcompute_convergence(min_scaling_factor,month)
if compute_convergence:
    for month in range(1,13):
        print(month)
        # Get the minimum scaling factor for this month
        min_scaling_factor = min_scaling_factors[month]
        print(min_scaling_factor)
        fcompute_convergence(min_scaling_factor,month,df15,df_SOC_min,df_cars)

## plot convergence

# Define the model function
def model_func(x, a, b):
    return a * np.power(x, -abs(b))

def plot_convergence(month):
    # Read the CSV file
    filename = 'convergence{}.csv'.format(month)
    df = pd.read_csv(filename)
    # Filter out the rows where df['Energy_coverage'] is positive
    df = df[df['scaling_factor'] <= 1000]
    df = df[df['Energy_coverage'] > 0]

    # Sort the DataFrame by the scaling factor column in ascending order
    df = df.sort_values(by='scaling_factor', ascending=True)
    # Fit a polynomial of degree 3 to the data
    coefficients = np.polyfit(df['scaling_factor'], df['Total_needed_max'], 3)

    # Create a polynomial function with these coefficients
    polynomial = np.poly1d(coefficients)

    # Generate y-values by evaluating the polynomial at the x-values
    y_values = polynomial(df['scaling_factor'])

    # Plot the original data as points
    plt.plot(df['scaling_factor'], df['Total_needed_max'], 'ro', ms=5)

    # Plot the polynomial curve
    plt.plot(df['scaling_factor'], y_values, 'b-')

    plt.xlabel('Scaling Factor')
    plt.ylabel('Total_needed_max')
    plt.title('Total_needed_max vs. Scaling Factor')
    plt.grid(True)
    plt.gca().invert_xaxis()  # This will invert the x-axis
    plt.show()

if plot_true:
    plot_convergence(month)
