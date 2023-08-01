import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from Scaling_mobility import scaledata, SOC_min_vect
recompute_dfNY_WIND= False
smart_charging_=False
TotalAllmonths_smart_charging=True
plotWindvsNY=False
plot_result=False
plotdiff=False
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
scaling=10
dfmobility= pd.read_csv('expanded_dataset.csv', index_col='time_interval')
energy_loss_driving = 0.75




#### scaled data :
recomputescaled=False
if recomputescaled==True :
    df_cars=scaledata(dfmobility)
    df_cars.to_csv(f'{scaling}SyntheticData.csv')
    logging.info('done creating dataset')
if recomputescaled==False:
    df_cars=pd.read_csv(f'{scaling}SyntheticData.csv', index_col='time_interval')

# Apply the function to each column of the DataFrame
SOC_min_recompute=False
if SOC_min_recompute==True:
    df_soc_min = df_cars.apply(SOC_min_vect)
    df_soc_min.to_csv(f'{scaling}_scaleSOCmin_data.csv')
    logging.info('done creating SOC MIN  dataset')

if SOC_min_recompute==False:
    df_SOC_min=pd.read_csv(f'{scaling}_scaleSOCmin_data.csv', index_col='time_interval', parse_dates=True)



battery_capacity = 70  # kWh
delta_t=0.25 #15 minutes
energy_loss_driving = 0.75    # energy loss per 15 minutes of driving
energy_loss_driving_winter=1
scenarios = [
    {"name": "scenario2_1", "capacity": 9, "charging_rate": 50, "discharging_rate": 20},
    {"name": "scenario1_1", "capacity": 13, "charging_rate": 50, "discharging_rate": 20},
    #{"name": "scenario1_2", "capacity": 9, "charging_rate": 50, "discharging_rate": 20},
    #{"name": "scenario1_3", "capacity": 9, "charging_rate": 50, "discharging_rate": 50},
    #{"name": "scenario2_1", "capacity": 13, "charging_rate": 50, "discharging_rate": 50},
    #{"name": "scenario2_2", "capacity": 2.5, "charging_rate": 19, "discharging_rate": 10},
    #{"name": "scenario3_1", "capacity": 4, "charging_rate": 50, "discharging_rate": 22},
    #{"name": "scenario3_2", "capacity": 2.5, "charging_rate": 50, "discharging_rate": 22}
]
monthplot=8
month=8

sample_syn=scaling
sample_size=sample_syn*2694


# original
#df_cars = pd.read_csv('expanded_dataset.csv', index_col='time_interval')
#df_cars = pd.read_csv('5SyntheticData.csv', index_col='time_interval')

df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')
#f_cars = df_cars.drop('total',axis=1)

#df_cars = pd.read_csv('2SyntheticJanuaryData.csv', index_col='time_interval')
for scenario in scenarios:
    print(f"Processing {scenario['name']}")
    capacity = scenario['capacity']
    charging_rate = scenario['charging_rate']
    discharging_rate = scenario['discharging_rate']
    #df = pd.read_csv(f'USWINDDATA_process_{capacity}M.csv', parse_dates=['Local_time_New_York'], delimiter=';')

    #df = pd.read_csv('USWINDDATA_process2.csv',parse_dates=['Local_time_New_York'], delimiter=';')
    #df_cars = pd.read_csv('Private_NbofEvs_1.csv', index_col='time_interval',parse_dates=True)
    #df_cars = df_cars.drop('total',axis=1)

    #df_NY = pd.read_csv('Actual_Load_NY.csv', parse_dates=['RTD End Time Stamp'], delimiter=';')

    #original data
    #df_SOC_min = pd.read_csv('SOCmin_data2.csv'  , index_col='time_interval',parse_dates=True)
    #df_SOC_min = pd.read_csv('2_scaleSOCmin_data.csv', index_col='time_interval', parse_dates=True)
    #df_SOC_min = pd.read_csv('5SyntheticData.csv', index_col='time_interval', parse_dates=True)
    #df_SOC_min = df_SOC_min.drop('total',axis=1)

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
        df15.to_csv(f'Wind_NY_Power{capacity}.csv')

    #if recompute_dfNY_WIND:
        #dfNY_WIND(df,df_NY,capacity)
        #df15 = pd.read_csv(f'Wind_NY_Power{capacity}.csv', index_col=0, parse_dates=True)
    if ~recompute_dfNY_WIND:
        df15 = pd.read_csv(f'Wind_NY_Power{capacity}.csv', index_col=0, parse_dates=True)


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

    def plot_diff(df1,df2,df3,df4,df5,df6,df7,month):
        df1 = df1[df1.index.month == month].iloc[1:]
        df2 = df2[df2.index.month == month].iloc[1:]
        df3 = df3[df3.index.month == month].iloc[1:]
        df4 = df4[df4.index.month == month].iloc[1:]
        df5 = df5[df5.index.month == month].iloc[1:]
        df6 = df6[df6.index.month == month].iloc[1:]
        df7 = df7[df7.index.month == month].iloc[1:]

        Pdifference1 = df1['Pdifference'].interpolate(method='cubic')
        Pdifference2 = df2['Pdifference'].interpolate(method='cubic')
        Pdifference3 = df3['Pdifference'].interpolate(method='cubic')
        Pdifference4 = df4['Pdifference'].interpolate(method='cubic')
        Pdifference5 = df5['Pdifference'].interpolate(method='cubic')
        Pdifference6 = df6['Pdifference'].interpolate(method='cubic')
        Pdifference7 = df7['Pdifference'].interpolate(method='cubic')
        # Create the figure and axes objects
        fig, ax = plt.subplots(figsize=(16, 10))

        # Plot the curves
        #ax.plot(df1.index, Pdifference1, color='red', label=f'Capacity {1*1000}')
        #ax.plot(df2.index, Pdifference2, color='blue', label=f'Capacity {2.5*1000}')
        #ax.plot(df3.index, Pdifference3, color='black', label=f'Capacity {4*1000}')
        ax.plot(df4.index, abs(Pdifference4), color='purple', label=f'Capacity {9*1000}')
        ax.plot(df5.index, abs(Pdifference5), color='green', label=f'Capacity {13*1000}')
        #ax.plot(df6.index, abs(Pdifference6), color='orange', label=f'Capacity {27*1000}')

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
    def results(df_CarCounts,df_results):
        df_results['Nmissing_total']=scaling_factor*df_CarCounts['Missing_noscale'] /(1-df_CarCounts['P_Blocked_high']-df_CarCounts['P_Blocked_low'])
        df_results['Ncharging_total']=scaling_factor*df_CarCounts['Charging_noscale']
        df_results['Ndischarging_total']=scaling_factor*df_CarCounts['Discharging_noscale']
        df_results['Ncritical']=scaling_factor*df_CarCounts['Critical_noscale']
        df_results['Total']=(df_results['Nmissing_total']+  df_results['Ncharging_total'] +df_results['Ndischarging_total']+df_results['Ncritical'])/(1-df_CarCounts['P_Blocked_high']-df_CarCounts['P_Blocked_low']-df_CarCounts['P_driving'])

        df_results['Ndriving']=df_CarCounts['P_driving']* df_results['Total']
        df_results['Nblocked']=(df_CarCounts['P_Blocked_high']+df_CarCounts['P_Blocked_low'])* df_results['Total']
        df_results.to_csv(f'RESULTS_month{month}_W{capacity}_alpha_{scaling_factor}_SOC_discharging_{discharging_rate}.csv')


    #def critical_discharge(t,parked_prev, condition, discharging_rate, battery_capacity, df_SOC,df_Power):
        #df_SOC.loc[t, ~condition & parked_prev] = df_SOC.shift().loc[t, ~condition & parked_prev] - (
                #discharging_rate * delta_t) / battery_capacity
        #df_Power.loc[t, ~condition & parked_prev] = (discharging_rate * scaling_factor) / 1000
        #df_CarCounts.loc[t, 'noscale'] = ~condition.sum()
    #def critical_charge(t,parked_prev, condition, charging_rate, battery_capacity, df_SOC,df_Power):
        #df_SOC.loc[t, ~condition & parked_prev] = df_SOC.shift().loc[t, ~condition & parked_prev] + (
                #charging_rate * delta_t) / battery_capacity
        #df_Power.loc[t, ~condition & parked_prev] = -(charging_rate * scaling_factor) / 1000

    def critical_SOC_power_calculator(t,parked_prev, critical, rate, battery_capacity, df_SOC,df_Power):
        df_SOC.loc[t, critical & parked_prev] = df_SOC.shift().loc[t, critical & parked_prev] + (rate*delta_t)/battery_capacity
        df_Power.loc[t, critical & parked_prev] = -(rate * scaling_factor)/1000
        #df_SOC.loc[t, critical & parked_prev] = df_SOC.shift().loc[t, critical & parked_prev]
        #df_Power.loc[t, critical & parked_prev] = 0
        df_CarCounts.loc[t, 'Critical'] = critical.sum()* scaling_factor

    def charge(t, condition2,parked_prev, critical, Pdifference_left, rate, battery_capacity, df_SOC,df_Power):
        parked_not_critical = parked_prev & ~critical

        # update cars that cannot be charged because soc is too high--> called the blocked cars
        df_SOC.loc[t, ~condition2 & ~critical& parked_prev] = df_SOC.shift().loc[t, ~condition2 & ~critical& parked_prev]
        df_Power.loc[t, ~condition2 & ~critical& parked_prev] =0
        ## sort SOC of remaining cars in ascending order
        SOC_sorted = df_SOC.shift().loc[t, condition2 & ~critical& parked_prev].sort_values(ascending=True)
        new_Power = pd.Series((-rate * scaling_factor) / 1000, index=SOC_sorted.index)
        cumulative_Power = new_Power.cumsum()
        # Determine the number of cars for which we need to calculate the Power
        Nb = cumulative_Power[-cumulative_Power < abs(Pdifference_left)].count()
        df_CarCounts.loc[t, 'Charging'] = Nb*scaling_factor
        #print('Nbcharge(t+1)',Nb)
        #print('lenSOC(t+1)',len(SOC_sorted))
        # Update SOC and Power for the cars that will be charging
        df_SOC.loc[t, new_Power.index[:Nb]] = df_SOC.shift().loc[t, new_Power.index[:Nb]] + (rate*delta_t) / battery_capacity
        df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]
        ##update soc and power for all remaining cars
        df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]
        # Update df_Power for car_ids in 'result'
        df_Power.loc[t, new_Power.index[Nb:]] = 0
        ##this is only to check if there are not enough of the normal cars available than we should charge the rest
        if Nb >= len(SOC_sorted):
            #print('fail Nb charge bigger than available')
            needed = math.ceil(abs(Pdifference_left) * 1000 / (rate * scaling_factor))
            logging.info('not working')
            logging.info(needed*scaling_factor)
            missing = needed - len(SOC_sorted)
            #print('missing_cars(t)', missing)
            df_CarCounts.loc[t, 'Missing_noscale'] = missing
            df_CarCounts.loc[t, 'Power_missing'] = -missing * (rate * scaling_factor) / 1000
        else:
            df_CarCounts.loc[t, 'Missing_noscale'] = 0
            #print('fail Nb charge bigger than available')
            #crit_charge_rate = 6
            #df_SOC.loc[t, parked_prev & ~condition2 & ~critical] = np.minimum(1, df_SOC.shift().loc[
                #t, parked_prev & ~condition2 & ~critical] + (crit_charge_rate * delta_t) / battery_capacity)
            #df_Power.loc[t, parked_prev & ~condition2 & ~critical] = -(crit_charge_rate * scaling_factor) / 1000



    def discharge(t, condition, parked_prev, critical, Pdifference_left, rate, battery_capacity, df_SOC, df_Power):
        # Sort the SOC by highest to lowest
        # Sort the SOC by highest to lowest
        parked_not_critical = parked_prev & ~critical
        # update SOC of all the cars with too low SOC for discharge--> blocked cars
        df_SOC.loc[t, ~condition & ~critical&parked_prev] = df_SOC.shift().loc[t,~condition& ~critical&parked_prev]
        df_Power.loc[t, ~condition& ~critical&parked_prev ] =0

        #available_discharge = (df_SOC.shift().loc[t,parked_prev] != 0 & ~critical)
        SOC_sorted = df_SOC.shift().loc[t, condition & ~critical & parked_prev].sort_values(ascending=False)
        # Calculate new Power for all parked, not critical cars
        # power given in KW we want it in MW need to dived by 1000 to get MW
        new_Power = pd.Series((rate * scaling_factor) / 1000, index=SOC_sorted.index)
        cumulative_Power = new_Power.cumsum()

        # Determine the number of cars for which we need to calculate the Power
        Nb = cumulative_Power[cumulative_Power <= abs(Pdifference_left)].count() + 1
        if Nb >= len(SOC_sorted):
            #print('fail Nb discharge bigger than available')
            needed=math.ceil(abs(Pdifference_left)*1000/(rate*scaling_factor))
            logging.info('not working')
            logging.info(needed*scaling_factor)
            missing=needed-len(SOC_sorted)
            df_CarCounts.loc[t, 'Power_missing'] = missing*(rate*scaling_factor)/1000
            #print('missing_cars(t)',missing)
            df_CarCounts.loc[t, 'Missing_noscale'] = missing
        else:
            missing=0
            df_CarCounts.loc[t, 'Missing_noscale'] = missing
            df_CarCounts.loc[t, 'Power_missing'] = 0
            #print('missing_cars(t)',missing)
        #print('Nbdischarge(t+1)',Nb)
        #print('lenSoCdischarge(t+1)',len(SOC_sorted))
        #print('indexdischarge',new_Power.index[:Nb])
        df_CarCounts.loc[t, 'Discharging'] = Nb* scaling_factor
        # Update SOC and Power for the cars that will be discharging
        df_SOC.loc[t, new_Power.index[:Nb]] = df_SOC.shift().loc[t, new_Power.index[:Nb]] - (rate*delta_t) / battery_capacity



        df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]
        # update SOC for all cars that are not discharging.
        df_SOC.loc[t, new_Power.index[Nb:]] = df_SOC.shift().loc[t, new_Power.index[Nb:]]
        # Update df_Power for car_ids in 'result'
        df_Power.loc[t, new_Power.index[Nb:]] = 0


    def charge_update_car_counts(t,condition2,parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size):
        df_CarCounts.loc[t, 'Charging'] = ((df_Power.loc[
                                               t] < 0) & ~critical).sum() * scaling_factor  # Negative power means charging
        df_CarCounts.loc[t, 'Discharging'] = (df_Power.loc[
                                                 t] > 0 ).sum() * scaling_factor  # Positive power means discharging## Sort the SOC by lowest to highest
        df_CarCounts.loc[
           t, 'Critical'] = critical.sum() * scaling_factor  # Already defined critical as a boolean Series, so sum gives count of True values#parked_not_critical = parked & ~critical
        df_CarCounts.loc[t, 'Driving'] = driving.sum()
        df_CarCounts.loc[t, 'Parked_neutral'] = parked.sum() - (
                   df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] + df_CarCounts.loc[
               t, 'Critical']) / scaling_factor
        df_CarCounts.loc[t, 'Notavailbutparked'] = (parked & ~condition2 & ~critical).sum()
        df_CarCounts.loc[t, 'Notavailbutparked'] = 0
        df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] +
                                              df_CarCounts.loc[t, 'Critical']) / (
                                                         1 - df_CarCounts.loc[t, 'Driving'] / sample_size-df_CarCounts.loc[t, 'Notavailbutparked'] / sample_size)
        df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] +
                                               df_CarCounts.loc[t, 'Critical']) / (
                                                          1 - df_CarCounts.loc[t, 'Driving'] / sample_size)

        df_CarCounts.loc[t, 'margin']=df_CarCounts.loc[t, 'Parked_neutral'] - df_CarCounts.loc[t, 'Notavailbutparked']

        df_CarCounts.loc[t, 'noscale']=(df_CarCounts.loc[t, 'Charging']+df_CarCounts.loc[t, 'Discharging']+df_CarCounts.loc[
           t, 'Critical'])/scaling_factor


    def discharge_update_car_counts(t,condition, parked, critical, driving, df_Power, df_CarCounts, df_cars, sample_size):
        df_CarCounts.loc[t, 'Charging'] = ((df_Power.loc[
                                                t] < 0) & ~critical).sum() * scaling_factor  # Negative power means charging
        df_CarCounts.loc[t, 'Discharging'] = (df_Power.loc[
                                                  t] > 0 ).sum() * scaling_factor  # Positive power means discharging## Sort the SOC by lowest to highest
        df_CarCounts.loc[
            t, 'Critical'] = critical.sum() * scaling_factor  # Already defined critical as a boolean Series, so sum gives count of True values#parked_not_critical = parked & ~critical
        df_CarCounts.loc[t, 'Driving'] = driving.sum()
        df_CarCounts.loc[t, 'Parked_neutral'] = parked.sum() - (
                    df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] + df_CarCounts.loc[
                t, 'Critical']) / scaling_factor
        df_CarCounts.loc[t, 'Notavailbutparked'] = (parked & ~condition & ~critical).sum()
        #df_CarCounts.loc[t, 'Notavailbutparked'] = 0
        df_CarCounts.loc[t, 'margin'] = df_CarCounts.loc[t, 'Parked_neutral'] - df_CarCounts.loc[t, 'Notavailbutparked']
        df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] +
                                               df_CarCounts.loc[t, 'Critical']) / (
                                                          1 - df_CarCounts.loc[t, 'Driving'] / sample_size -
                                                          df_CarCounts.loc[t, 'Notavailbutparked'] / sample_size)
        #df_CarCounts.loc[t, 'Total_needed'] = (df_CarCounts.loc[t, 'Charging'] + df_CarCounts.loc[t, 'Discharging'] +
                                               #df_CarCounts.loc[t, 'Critical']) / (
                                                      #1 - df_CarCounts.loc[t, 'Driving'] / sample_size)

        df_CarCounts.loc[t, 'noscale']=(df_CarCounts.loc[t, 'Charging']+df_CarCounts.loc[t, 'Discharging']+df_CarCounts.loc[
            t, 'Critical'])/scaling_factor

    def smart_charging(df_cars_shifted,df_cars,df_SOC,df_SOC_min_month,df_Power,df_15,df_CarCounts,month,scaling_factor,TotalAllmonths_smart_charging,charging_rate,discharging_rate):
        # Iterate over each time step
        for t in df_cars.index[1:]:
            logging.info(t)
            parked_prev = df_cars_shifted.loc[t] == 1
            driving_prev = ~parked_prev
            parked= df_cars.loc[t] == 1
            driving = ~parked
            df_CarCounts.loc[t, 'Driving'] = driving_prev.sum()
            #print('driving(t)',driving_prev.sum())
            #print('parked(t)',parked_prev.sum())
            #a=df_SOC_min_month.loc[t]

            df_SOC.loc[t, driving_prev] = df_SOC.shift().loc[t, driving_prev] - energy_loss_driving/battery_capacity
            df_Power.loc[t, driving_prev] = 0
            critical = ((df_SOC.shift().loc[t,parked_prev]-(discharging_rate*delta_t)/battery_capacity)<= df_SOC_min_month.loc[t,parked_prev]) & (df_cars.loc[t,parked_prev]==0)
            #print('critical(t)',critical.sum())
            #driving_prev=df_cars_shifted.loc[t] ==0
            # Update SOC and Power for driving cars
            critical_SOC_power_calculator(t, parked_prev, critical, charging_rate, battery_capacity, df_SOC, df_Power)
            Pagg_critical = df_Power.loc[t, critical& parked_prev].sum()
            Pdifference_left = (df_15.shift().loc[t, 'Pdifference']) + Pagg_critical
            #df_CarCounts.loc[t, 'Pdifference'] = df_15.shift().loc[t, 'Pdifference']
            df_CarCounts.loc[t, 'Pdifference_left']=Pdifference_left
            #Pdifference_left = df_15.loc[t, 'Pdifference'] + Pagg_critical

             # preventive charging and dicharging so that the SOC does not converge too fast
            # if we do this since the soc converges fast, the all charge at the same time.
            #condition2 = df_SOC.shift().loc[t] <= 1 - (charging_rate * delta_t) / battery_capacity
            #condition = df_SOC.shift().loc[t] > (discharging_rate * delta_t) / battery_capacity

            #critical_discharge(t, parked_prev, condition2, discharging_rate, battery_capacity, df_SOC, df_Power)
            #Pagg_critical_discharge = df_Power.loc[t, ~condition2 & parked_prev].sum() # this is positive since dicharge
            #critical_charge(t, parked_prev, condition, charging_rate, battery_capacity, df_SOC, df_Power)
            #Pagg_critical_charge = df_Power.loc[t, ~condition & parked_prev].sum()
            #Pdifference_left = df_15.loc[t, 'Pdifference'] + Pagg_critical #+Pagg_critical_charge  #Pagg_critical_discharge

            # If there's remaining power left after charging critical cars, proceed with parked cars
            if Pdifference_left > 0:
                #print('Pdifference_left',Pdifference_left)
                #condition2 = df_SOC.shift().loc[t,parked_prev ] <= 1 - (charging_rate * delta_t) / battery_capacity
                #print('condition2_blocked(t-1)', (~condition2).sum())
                condition2 = df_SOC.shift().loc[t,parked_prev & ~critical] <= 1 - (charging_rate * delta_t) / battery_capacity
                df_CarCounts.loc[t, 'Blocked_high'] = (~condition2).sum()
                if df_CarCounts.loc[t, 'Driving']+df_CarCounts.loc[t, 'Blocked_high']>sample_size:
                    logging.info('system fails')
                    break
                df_CarCounts.loc[t, 'P_Blocked_high'] = (~condition2).sum()/sample_size
                #print('condition2_blocked_try(t)',(~condition2).sum())
                #df_CarCounts.loc[t, 'SOC_high'] = (parked & ~condition2 ).sum
                #df_CarCounts.loc[t, 'SOC_low']=0
                charge(t, condition2,parked_prev, critical, Pdifference_left, charging_rate, battery_capacity, df_SOC,
                       df_Power)
                df_CarCounts.loc[t, 'Total'] =(df_CarCounts.loc[t, 'Charging']+df_CarCounts.loc[t, 'Critical'])/(1-df_CarCounts.loc[t, 'Driving']/sample_size-df_CarCounts.loc[t, 'P_Blocked_high'])
                logging.info(df_CarCounts.loc[t, 'Total'])
                #charge_update_car_counts(t, condition2, parked, critical, driving, df_Power, df_CarCounts, df_cars,
                                         #sample_size)
            if Pdifference_left < 0:
                #scaling_factor=1500
                #print('Pdifference_left', Pdifference_left)
                #conditiontry = df_SOC.shift().loc[t,parked_prev ] > (discharging_rate * delta_t) / battery_capacity
                #print('condition_blockedtry(t)',(~conditiontry).sum())

                condition = df_SOC.shift().loc[t,parked_prev & ~critical] > (discharging_rate * delta_t) / battery_capacity
                df_CarCounts.loc[t, 'Blocked_low'] = (~condition).sum()
                df_CarCounts.loc[t, 'P_Blocked_low'] = (~condition).sum() / sample_size
                if df_CarCounts.loc[t, 'Driving']+df_CarCounts.loc[t, 'Blocked_low']>sample_size:
                    logging.info('system fails')
                    break

                discharge(t, condition, parked_prev, critical, Pdifference_left, discharging_rate, battery_capacity,
                          df_SOC, df_Power)
                df_CarCounts.loc[t, 'Total'] =(df_CarCounts.loc[t, 'Discharging']+df_CarCounts.loc[t, 'Critical'])/(1-df_CarCounts.loc[t, 'Driving']/sample_size-df_CarCounts.loc[t, 'P_Blocked_low'])
                logging.info(df_CarCounts.loc[t, 'Total'])
                #discharge_update_car_counts(t,condition, parked, critical, driving, df_Power, df_CarCounts, df_cars,
                                            #sample_size)

        #df_CarCounts=df_CarCounts.shift(1)
        df_15_shift=df_15.shift(1)
        df_Power_sum_ = df_Power.sum(axis=1).to_frame().rename(columns={0: 'PV2G'})
        df_Power_sum_1  = df_15_shift.join(df_Power_sum_)
        df_Power_sum_month = df_Power_sum_1.join(df_CarCounts)
        df_Power_sum_month['P_balance_without'] = df_Power_sum_month['PV2G'] + df_15['Pdifference']
        df_Power_sum_month['P_balance_with_missing'] = df_Power_sum_month['PV2G'] + df_15['Pdifference']+df_Power_sum_month['Power_missing']

        #the first row needs to be removed because the cars are not actively participating in V2G.
        #df_Power_sum_month = df_Power_sum_month.iloc[1:]
        df_SOC.to_csv(f'NEW_{scaling}_SOC_month{month}_W{capacity}_alpha_{scaling_factor}_SOC_discharging_{discharging_rate}.csv')
        df_Power_sum_month.to_csv(f'NEW_{scaling}Power_month{month}_W{capacity}alpha_{scaling_factor}_discharging_{discharging_rate}.csv')

        # Store the dataframe in the dictionary with the month as the key
        dfs[f"df_Power_sum_{month}"] = df_Power_sum_month
        TotalAllmonths_smart_charging=False
        if TotalAllmonths_smart_charging:
            result = dfs[f'df_Power_sum_{month}']['Total_needed'].max()
            margin = dfs[f'df_Power_sum_{month}']['margin'].min()
            max_total_index = dfs[f'df_Power_sum_{month}']['Total_needed'].idxmax()
            PV2G = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'PV2G']
            Pdiff = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'Pdifference']
            rmse = np.sqrt((dfs[f'df_Power_sum_{month}']['P_balance'] ** 2).mean())
            Nb = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'noscale']
            p1 = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'Driving'] / sample_size
            p2 = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'Notavailbutparked'] / sample_size
            Nb4 = Nb / (1 - p1 - p2)
            Nb2 = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'Driving'] / sample_size * Nb4
            Nb3 = dfs[f'df_Power_sum_{month}'].loc[max_total_index, 'Notavailbutparked'] / sample_size * Nb4
            return result, margin, Nb, Nb2, Nb3, Nb4, p1, p2, PV2G, rmse, Pdiff



    #min_scaling_factors = {
    #    1:255,
    #    2:245,
    #    3:245,
    #    4:215,
    #    5:330,
    #    6:325,
    #    7:360,
    #    8:1000,
    #    9:310,
    #    10:310,
    #    11:230,
    #    12:250
    #}
    min_scaling_factors = {
        1:500,
        2:200,
        3:200,
        4:300,
        5:400,
        6:200,
        7:400,
        8:400,
        9:200,
        10:200,
        11:200,
        12:200
    }
    if smart_charging_:
        dfs = {}
        TotalAllmonths_smart_charging=False
        df_cars = load_and_extract_month(df_cars, month)
        df_15 = load_and_extract_month(df15, month)
        df_SOC_min = load_and_extract_month(df_SOC_min, month)

        # Initialize randomly the SOC for each car
        df_SOC = pd.DataFrame(
            np.random.choice(np.linspace(0.4, 1, 10), size=df_cars.shape),
            index=df_cars.index,
            columns=df_cars.columns
        )
        df_Power = pd.DataFrame(0, index=df_cars.index, columns=df_cars.columns)
        df_CarCounts = pd.DataFrame(0, index=df_cars.index, columns=['Charging', 'Discharging', 'Critical', 'Driving'])
        df_SOC_shifted = df_SOC.shift()
        df_SOC_min_shifted = df_SOC_min.shift()
        df_cars_shifted=df_cars.shift()
        scaling_factor = min_scaling_factors[month]
        smart_charging(df_cars_shifted,df_cars,df_15,df_SOC,df_SOC_min,month,scaling_factor,TotalAllmonths_smart_charging)
    if plotWindvsNY:
        plotWind_NY(df15,monthplot)
    if plot_result:
        scaling_factor = min_scaling_factors[month]
        plotWind_NY_V2G(capacity,scaling_factor,month,discharging_rate,charging_rate)

    if TotalAllmonths_smart_charging:
        # Create an empty dictionary to store the dataframes
        dfs = {}
        Total_month = pd.DataFrame(columns=['month', 'scaling_factor', 'Total_needed_max', 'Energy_coverage'])

        for month in range(1,13):
            logging.info(month)
            df_cars_month=load_and_extract_month(df_cars,month)
            df_15=load_and_extract_month(df15,month)
            df_SOC_min_month = load_and_extract_month(df_SOC_min, month)
            # Initialize randomly the SOC for each car
            df_SOC = pd.DataFrame(
                np.random.choice(np.linspace(0.1, 1, 10), size=df_cars_month.shape),
                index=df_cars_month.index,
                columns=df_cars_month.columns
            )
            df_Power = pd.DataFrame(0, index=df_cars_month.index, columns=df_cars_month.columns)
            df_CarCounts = pd.DataFrame(0, index=df_cars_month.index, columns=['Pdifference_left','Power_missing','Charging', 'Discharging', 'Critical', 'Driving','Missing_noscale','Blocked_high','Blocked_low','P_Blocked_high','P_Blocked_low','P_driving'])
            df_results = pd.DataFrame(0, index=df_cars_month.index,
                                      columns=['Total', 'Nmissing_total', 'Ncharging_total', 'Ndischarging_total',
                                               'Ncritical', 'Ndriving', 'Nblocked'])
            #df_Power_sum = pd.DataFrame(0,index=df_cars.index,columns=['Power_output','NYC_demand','Pdifference','Charging', 'Discharging', 'Critical', 'Driving','Total_needed','P_balance'])
            #df_SOC_shifted = df_SOC.shift()
            df_SOC_min_shifted = df_SOC_min_month.shift()
            df_cars_shifted=df_cars_month.shift()
            scaling_factor = min_scaling_factors[month]
            logging.info(scaling_factor)
            result,margin, Nb, Nb2, Nb3,Nb4, p1, p2,PV2G,rmse,Pdiff=smart_charging(df_cars_shifted,df_cars_month,df_SOC,df_SOC_min_month,df_Power,df_15,df_CarCounts,month,scaling_factor,TotalAllmonths_smart_charging,charging_rate,discharging_rate)
            # Create a dictionary with column names as keys and values
            #row_data = {
            #    'month': month,
            #    'scaling_factor': scaling_factor,
            #    'Total_needed_max': result,
            #    'Energy_coverage': margin,
            #    'PV2G': PV2G,
            #    'Pdiff': Pdiff,
            #    'Pbalance_rmse': rmse,
            #    'Nb_indivual_cars': Nb,
            #    'Nb_driving': Nb2,
            #    'Nb_parked': Nb3,
            #    'prob_driving': p1,
            #    'prob_parking': p2,
            #    'DatasetNbcars': Nb4
            #}
            ## Append the dictionary as a new row to the dataframe
            #Total_month = Total_month.append(row_data, ignore_index=True)

        #Total_month.to_csv(f'/Users/louisahillegaart/pythonProject1/RESULTS/{scenario["name"]}/total_all_month_{capacity}M.csv', index=False)


    if plotdiff:
        lcapacity = [1, 2.5, 4, 9,13,27,140]
        df_capacity = {}  # Dictionary to store data frames with capacity values
        for j in lcapacity:
            filename = f'Wind_NY_Power{j}'
            df_capacity[j] = pd.read_csv(filename, index_col=0, parse_dates=True)

        plot_diff(*df_capacity.values(), month)


    
