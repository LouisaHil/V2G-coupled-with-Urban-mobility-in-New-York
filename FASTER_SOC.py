import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

recompute_dfNY_WIND= True
smart_charging_=False
TotalAllmonths_smart_charging=False
plotWindvsNY=True
plot_result=False
plotdiff=False

monthplot=1
month=8
#capacity=4 # 4000 MW
#capacity=13 # 13000 MW
#capacity=2.5 # 2500 MW
capacity=220
#capacity=1 1000MW
#capacity=9 #9000MW
sample_syn=1
sample_size=sample_syn*2694

#scaling_factor = 700
battery_capacity = 70  # kWh
charging_rate = 50 / 4  # kW (charging for 15 minutes)
discharging_rate = 50 / 4
energy_loss_driving = 0.75    # energy loss per 15 minutes of driving

# original
df_cars = pd.read_csv('expanded_dataset.csv', index_col='time_interval')
df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')
df_cars = df_cars.drop('total',axis=1)

#df_cars = pd.read_csv('2SyntheticJanuaryData.csv', index_col='time_interval')

df = pd.read_csv(f'USWINDDATA_process_{capacity}M.csv', parse_dates=['Local_time_New_York'], delimiter=';')
#df = pd.read_csv('USWINDDATA_process2.csv',parse_dates=['Local_time_New_York'], delimiter=';')
#df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col='time_interval',parse_dates=True)
#df_cars = df_cars.drop('total',axis=1)

df_NY = pd.read_csv('Actual_Load_NY.csv', parse_dates=['RTD End Time Stamp'], delimiter=';')

#original data
df_SOC_min = pd.read_csv('SOCmin_data2.csv'  , index_col='time_interval',parse_dates=True)
df_SOC_min = df_SOC_min.drop('total',axis=1)

# for synthetic data you need to uncomment this
#df_SOC_min = pd.read_csv('2_scaleSOCmin_data.csv', index_col='time_interval',parse_dates=True)


#df_SOC_min = df_SOC_min[df_SOC_min.index.month == month]
def preprocess_wind(df):
    df = df.sort_values('Local_time_New_York')
    start_date = df['Local_time_New_York'].min()
    end_date = df['Local_time_New_York'].max()
    freq = '1H'  # 15 minutes
    complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    # Step 3: Find missing datetimes
    missing_indices = []
    expected_datetime = start_date
    for index, row in df.iterrows():
        if row['Local_time_New_York'] != expected_datetime:
            missing_indices.append(index)
            df.at[index, 'Local_time_New_York'] = expected_datetime
        expected_datetime += pd.Timedelta(hours=1)
    df.set_index('Local_time_New_York', inplace=True)
    df = df.drop(['Time_UTC', 'wind_speed'], axis=1)
    df_interpolated = df.resample('15T').interpolate()
    new_index = pd.DatetimeIndex([date.replace(year=2013) for date in df_interpolated.index])
    df_interpolated.index = new_index
    ##### we need to add three rows for the time 23:15, 23:30, 23:45 of the last day of the year
    new_rows = pd.DataFrame([df_interpolated.iloc[-1]] * 3, columns=df_interpolated.columns)
    new_rows.index = pd.date_range(df_interpolated.index[-1], periods=4, freq='15T')[1:]
    df_interpolated = df_interpolated.append(new_rows)
    df_interpolated['Electricity'] = df_interpolated['Electricity'] / 1000
    df_interpolated = df_interpolated.rename(columns={'Electricity': 'power_output'})
    return df_interpolated

def preprocess_NY(df_NY):
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
    new_index = pd.DatetimeIndex([date.replace(year=2013) for date in df_NY.index])
    df_NY.index = new_index
    return df_NY


def dfNY_WIND(df_wind,df_NY,capacity):
    df15=preprocess_wind(df_wind)
    df_NY=preprocess_NY(df_NY)
    df15['NYC_demand'] = df_NY['RTD Actual Load'].values
    df15['Pdifference'] = df15['power_output']-df15['NYC_demand']
    df15.to_csv(f'Wind_NY_Power{capacity}')


if recompute_dfNY_WIND:
    dfNY_WIND(df,df_NY,capacity)
    df15 = pd.read_csv(f'Wind_NY_Power{capacity}', index_col=0, parse_dates=True)
if ~recompute_dfNY_WIND:
    df15 = pd.read_csv(f'Wind_NY_Power{capacity}', index_col=0, parse_dates=True)


def load_and_extract_month(df,month):
    df = df[df.index.month == month]
    return df

def plotWind_NY(df,month):
    # Create the cubic curves
    df = df[df.index.month == month].iloc[1:]
    power_output_curve = df['power_output'].interpolate(method='cubic')
    nyc_demand_curve = df['NYC_demand'].interpolate(method='cubic')
    #Pdifference=df['Pdifference'].interpolate(method='cubic')

    # Plot the curves
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, power_output_curve, color='black', label=f'Wind Power Output with capacity: {capacity*1000}')
    plt.plot(df.index, nyc_demand_curve, color='blue', label='NYC Demand')
    #plt.plot(df.index, Pdifference, color='black', label='Power difference')
    # Fill the area between the curves
    plt.fill_between(df.index, power_output_curve, nyc_demand_curve, where=(power_output_curve > nyc_demand_curve),
                     color='green', alpha=0.5)
    plt.fill_between(df.index, power_output_curve, nyc_demand_curve, where=(nyc_demand_curve > power_output_curve),
                     color='gray', alpha=0.5)

    # Set labels and title
    plt.xlabel('Datetime')
    plt.ylabel('Values')
    plt.title('Power Output vs NYC Demand')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def plot_diff(df1,df2,df3,df4,df5,df6,month):
    df1 = df1[df1.index.month == month].iloc[1:]
    df2 = df2[df2.index.month == month].iloc[1:]
    df3 = df3[df3.index.month == month].iloc[1:]
    df4 = df4[df4.index.month == month].iloc[1:]
    df5 = df5[df5.index.month == month].iloc[1:]
    df6 = df6[df6.index.month == month].iloc[1:]

    Pdifference1 = df1['Pdifference'].interpolate(method='cubic')
    Pdifference2 = df2['Pdifference'].interpolate(method='cubic')
    Pdifference3 = df3['Pdifference'].interpolate(method='cubic')
    Pdifference4 = df4['Pdifference'].interpolate(method='cubic')
    Pdifference5 = df5['Pdifference'].interpolate(method='cubic')
    Pdifference6 = df6['Pdifference'].interpolate(method='cubic')
    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot the curves
    #ax.plot(df1.index, Pdifference1, color='red', label=f'Capacity {1*1000}')
    ax.plot(df2.index, Pdifference2, color='blue', label=f'Capacity {2.5*1000}')
    #ax.plot(df3.index, Pdifference3, color='black', label=f'Capacity {4*1000}')
    ax.plot(df4.index, Pdifference4, color='purple', label=f'Capacity {9*1000}')
    #ax.plot(df5.index, Pdifference5, color='orange', label=f'Power difference with capacity {13*100}')
    #ax.plot(df6.index, Pdifference6, color='orange', label=f'Power difference with capacity {140*100}')

    # Set labels and title
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Values')
    ax.set_title('Absolute value of Power difference between Wind and NYC demand from different Wind capacity factors')

    # Add legend
    ax.legend(fontsize=10)

    # Show the plot
    plt.show()



def plotWind_NY_V2G(capacity,scaling_factor,month,discharging_rate,charging_rate):
    filename = f'W{capacity}alpha_{scaling_factor}_Power{month}_discharging_{discharging_rate}.csv'
    df=pd.read_csv(filename)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0], inplace=True)
    power_output_curve = df['power_output'].interpolate(method='cubic')
    nyc_demand_curve = df['NYC_demand'].interpolate(method='cubic')
    PV2G=df['PV2G'].interpolate(method='cubic')
    TotalCars = math.ceil(df['Total_needed'].max())

    # Plot the curves
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(df.index, power_output_curve, color='purple', label='Wind Power Output')
    ax.plot(df.index, PV2G, color='green', label='Power V2G')
    #ax.plot(df.index, PV2G + power_output_curve, color='yellow', label='Power V2G+Windpower')
    ax.plot(df.index, nyc_demand_curve, color='grey', label='NYC Demand')

    # Set labels and title
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Power MW')
    ax.set_title(f'Wind Power Output and PV2G Power vs NYC Demand')


    # Set additional information text box on the left side
    additional_info = f'Month: {month}\nCharging Rate (Kwh): {charging_rate*4}\nDischarging Rate (Kwh): {discharging_rate*4}\nScaling Factor: {scaling_factor}\nWind Capacity(MW): {capacity*1000}\nTotal Cars: {TotalCars}'
    text_box = plt.text(0.01, 0.99, additional_info, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.05'))

    # Add legend on the top right
    ax.legend(loc='upper right')

    # Show the plot
    plt.show()

#synthetic_size=1

#df_cars = pd.read_csv('20SyntheticJanuaryData.csv', index_col='time_interval',parse_dates=True)
#df_cars = df_cars.drop('total',axis=1)


## synthetic data
#df_SOC_min = pd.read_csv('20BIGSOCmin_data.csv', index_col='time_interval',parse_dates=True)
##month
#df_SOC_min = pd.read_csv('SOCmin_data.csv', index_col='time_interval',parse_dates=True)



def critical_SOC_power_calculator(t, critical, rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power):
    df_SOC.loc[t, critical] = df_SOC_shifted.loc[t, critical] + rate/battery_capacity
    df_Power.loc[t, critical] = -(charging_rate * scaling_factor)/1000
    df_CarCounts.loc[t, 'noscale'] = critical.sum()

def charge(t, parked, critical, Pdifference_left, rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power):
    parked_not_critical = parked & ~critical
    SOC_sorted = df_SOC.shift().loc[t, parked_not_critical].sort_values(ascending=True)
    new_Power = pd.Series((-rate * scaling_factor) / 1000, index=SOC_sorted.index)
    cumulative_Power = new_Power.cumsum()
    # Determine the number of cars for which we need to calculate the Power
    Nb = cumulative_Power[-cumulative_Power < abs(Pdifference_left)].count()
    df_CarCounts.loc[t, 'noscale'] = Nb
    # Update SOC and Power for the cars that will be charging
    df_SOC.loc[t, new_Power.index[:Nb]] = np.minimum(1, df_SOC_shifted.loc[t, new_Power.index[:Nb]] + rate / battery_capacity)
    df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]
    df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    df_Power.loc[t, new_Power.index[Nb:]] = 0

def discharge(t, condition, parked, critical, Pdifference_left, rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power):
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
    df_CarCounts.loc[t, 'noscale'] = Nb
    # Update SOC and Power for the cars that will be discharging
    df_SOC.loc[t, new_Power.index[:Nb]] = df_SOC_shifted.loc[t, new_Power.index[:Nb]] - rate / battery_capacity
    df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]

    df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC_shifted.loc[t, new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    df_Power.loc[t, new_Power.index[Nb:]] = 0


def charge_update_car_counts(t,parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size):
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
    df_CarCounts.loc[t, 'margin']=df_CarCounts.loc[t, 'Parked_neutral'] - df_CarCounts.loc[t, 'Notavailbutparked']

    #

def discharge_update_car_counts(t,condition, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size):
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

def smart_charging(month,scaling_factor,TotalAllmonths_smart_charging):
    # Iterate over each time step
    for t in df_cars.index[1:]:
        print(t)
        parked = df_cars.loc[t] == 1
        driving = ~parked
        critical = (df_SOC_shifted.loc[t] <= df_SOC_min_shifted.loc[t]) & (df_cars.loc[t]==0)

        # Update SOC and Power for driving cars
        df_SOC.loc[t, driving] = df_SOC_shifted.loc[t, driving] - energy_loss_driving/battery_capacity
        df_Power.loc[t, driving] = 0

        # Check for Pdifference>0 or Pdifference <0
        if df15.loc[t, 'Pdifference'] > 0:  # Charging
            # Calculate SOC and Power for critical cars
            critical_SOC_power_calculator(t, critical, charging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power)
            Pagg_critical = df_Power.loc[t, critical].sum()
            Pdifference_left = df15.loc[t, 'Pdifference'] + Pagg_critical
            # If there's remaining power left after charging critical cars, proceed with parked cars
            if Pdifference_left > 0:
                charge(t, parked, critical, Pdifference_left, charging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power)
            if Pdifference_left < 0:
                condition = df_SOC_shifted.loc[t, parked] > discharging_rate / battery_capacity
                discharge(t,condition, parked, critical, Pdifference_left, discharging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power)
            charge_update_car_counts(t, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size)

        elif df15.loc[t, 'Pdifference'] < 0:  # Discharging
            # Calculate SOC and Power for critical cars
            critical_SOC_power_calculator(t, critical, charging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power)
            Pagg_critical = df_Power.loc[t, critical].sum()
            Pdifference_left = df15.loc[t, 'Pdifference'] + Pagg_critical
            condition = df_SOC_shifted.loc[t, parked] > discharging_rate / battery_capacity

            discharge(t,condition, parked, critical, Pdifference_left, discharging_rate, battery_capacity, df_SOC, df_SOC_shifted, df_Power)

            discharge_update_car_counts(t,condition, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size)

    df_Power_sum = df_Power.sum(axis=1).to_frame().rename(columns={0: 'PV2G'})
    df_Power_sum  = df15.join(df_Power_sum)
    df_Power_sum =df_Power_sum .join(df_CarCounts)
    df_SOC.to_csv(f'W{capacity}_alpha_{scaling_factor}_SOC{month}_discharging_{discharging_rate}.csv')
    df_Power_sum .to_csv(f'W{capacity}alpha_{scaling_factor}_Power{month}_discharging_{discharging_rate}.csv')
    if TotalAllmonths_smart_charging:
        result = df_CarCounts['Total_needed'].max()
        # median=df_CarCounts['Total_needed'].median()
        margin = df_CarCounts['margin'].min()
        # Find the index at which 'total' is maximized
        max_total_index = df_CarCounts['Total_needed'].idxmax()
        PV2G = df_Power_sum.loc[max_total_index, 'PV2G']
        Pdiff = df_Power_sum.loc[max_total_index, 'Pdifference']
        # Use this index to get the corresponding value in the 'charging' column return result,median,margin, df_Nb_cars_noscale
        df_CarCounts['P_balance'] = df_Power_sum['PV2G'] + df15['Pdifference']
        rmse = np.sqrt((df_CarCounts['P_balance'] ** 2).mean())
        Nb = df_CarCounts.loc[max_total_index, 'noscale']
        p1 = df_CarCounts.loc[max_total_index, 'Driving'] / sample_size
        p2 = df_CarCounts.loc[max_total_index, 'Notavailbutparked'] / sample_size
        Nb4 = Nb / (1 - p1 - p2)
        Nb2 = df_CarCounts.loc[max_total_index, 'Driving'] / sample_size * Nb4
        Nb3 = df_CarCounts.loc[max_total_index, 'Notavailbutparked'] / sample_size * Nb4
        return result,margin, Nb, Nb2, Nb3,Nb4, p1, p2,PV2G,rmse,Pdiff




min_scaling_factors = {
    1:255,
    2:245,
    3:245,
    4:215,
    5:330,
    6:325,
    7:360,
    8:1000,
    9:310,
    10:310,
    11:230,
    12:250
}
if smart_charging_:
    TotalAllmonths_smart_charging=False
    df_cars = load_and_extract_month(df_cars, month)
    df15 = load_and_extract_month(df15, month)
    df_SOC_min = load_and_extract_month(df_SOC_min, month)

    # Initialize randomly the SOC for each car
    df_SOC = pd.DataFrame(
        np.random.choice(np.linspace(0.1, 1, 10), size=df_cars.shape),
        index=df_cars.index,
        columns=df_cars.columns
    )
    df_Power = pd.DataFrame(0, index=df_cars.index, columns=df_cars.columns)
    df_CarCounts = pd.DataFrame(0, index=df_cars.index, columns=['Charging', 'Discharging', 'Critical', 'Driving'])
    df_SOC_shifted = df_SOC.shift()
    df_SOC_min_shifted = df_SOC_min.shift()
    scaling_factor = min_scaling_factors[month]
    smart_charging(month,scaling_factor,TotalAllmonths_smart_charging)
if plotWindvsNY:
    plotWind_NY(df15,monthplot)
if plot_result:
    scaling_factor = min_scaling_factors[monthplot]
    plotWind_NY_V2G(capacity,scaling_factor,monthplot,discharging_rate,charging_rate)

if TotalAllmonths_smart_charging:
    for month in range(1,13):
        print(month)
        df_cars=load_and_extract_month(df_cars,month)
        df_15=load_and_extract_month(df15,month)
        df_SOC_min = load_and_extract_month(df_SOC_min, month)

        # Initialize randomly the SOC for each car
        df_SOC = pd.DataFrame(
            np.random.choice(np.linspace(0.1, 1, 10), size=df_cars.shape),
            index=df_cars.index,
            columns=df_cars.columns
        )
        df_Power = pd.DataFrame(0, index=df_cars.index, columns=df_cars.columns)
        df_CarCounts = pd.DataFrame(0, index=df_cars.index, columns=['Charging', 'Discharging', 'Critical', 'Driving'])
        df_SOC_shifted = df_SOC.shift()
        df_SOC_min_shifted = df_SOC_min.shift()
        scaling_factor = min_scaling_factors[month]
        print(scaling_factor)
        result,margin, Nb, Nb2, Nb3,Nb4, p1, p2,PV2G,rmse,Pdiff=smart_charging(month,scaling_factor,TotalAllmonths_smart_charging)
        Total_month = pd.DataFrame(columns=['month','scaling_factor','Total_needed_max', 'Energy_coverage'])
        Total_month=Total_month.append(
                {'month':month,'scaling_factor': scaling_factor, 'Total_needed_max': result, 'Energy_coverage':margin,'PV2G':PV2G,'Pdiff':Pdiff, 'Pbalance_rmse':rmse,'Nb_indivual_cars':Nb, 'Nb_driving':Nb2,'Nb_parked':Nb3, 'prob_driving': p1, 'prob_parking':p2,'DatasetNbcars':Nb4},
                ignore_index=True)
        print(Total_month)
    Total_month.to_csv('total_all_month', index=False)


if plotdiff:
    lcapacity = [1, 2.5, 4, 9,13,140]
    df_capacity = {}  # Dictionary to store data frames with capacity values
    for j in lcapacity:
        filename = f'Wind_NY_Power{j}'
        df_capacity[j] = pd.read_csv(filename, index_col=0, parse_dates=True)

    plot_diff(*df_capacity.values(), monthplot)