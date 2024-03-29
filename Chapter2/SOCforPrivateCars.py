import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from itertools import accumulate
import concurrent.futures
from scipy.integrate import simps, quad
##variables :

SOCmax=1
#SOCmedian=0.65 # this should remain low if we want to reduce the strain on the grind when there is no wind
deltaEnergy=0.75 #kwh for 15 min drive assuming speed of 8.8 miles/hour
CB=70 #maximum battery capacity
#r_c=150 #rate of charge
#r_d=40 # rate of discharge
T_i= 15/60 # time interval between each measurement in hours
 # time of discharge that changed
#initialize time
#Energy_level=[CB] # full capacity
#SOC = [1] # initializing SOC at the beginning of January
count=0
#medallion = '2013000066'

## 2019 ## change here for new month
start_time = '2019-01-1 00:00:00'
start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = '2019-01-30 23:45:00'
end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

## 2013 ##change here for new month
start = datetime(2013, 1, 1, 0, 0)
end = datetime(2013, 1, 30, 23, 30)

## wind data
df = pd.read_csv('USWINDDATA_process.csv',parse_dates=['Local_time_New_York'], index_col='Local_time_New_York', delimiter=';')
NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])

# Read the data from cars: This is the dataset for january--> change is required here for different month or if we would like to run the algorithm for the entire year and filer for each month after
df_taxi = pd.read_csv('Private_NbofEvs_1.csv', index_col=0, parse_dates=True)

## definitions
def interval_dataframe(df,start_time,end_time):
    day = df.loc[start_time:end_time]
    time = day.index
    power_output=day['Electricity'].to_numpy()/1000
    return day, time, power_output

def fnum_days(start_time,end_time):
    start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    time_diff = end_datetime - start_datetime
    num_days = time_diff.days + 1  # Add one day to include January 1st
    return num_days

def NYdemand_days(num_days,NYC_demand):
    NYC_demand = np.tile(NYC_demand, num_days)
    return NYC_demand

def interpolation(num_days,NYC_demand,power_output):
    x=np.arange(0,24*num_days,1)
    xnew = np.linspace(0, 23*num_days, 1000)
    f = interpolate.interp1d(x, NYC_demand, kind='cubic')
    g=interpolate.interp1d(x, power_output, kind='cubic')
    NYC_demand_new = f(xnew)
    power_output_new=g(xnew)
    return xnew,NYC_demand_new, power_output_new

def intervals(xnew, num_days,NYC_demand_new,power_output_new):
    ## define the intersection points :
    diff = NYC_demand_new - power_output_new
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    x_intersections = xnew[sign_changes]
    x_intersections = x_intersections.tolist()  # Convert to list
    x_intersections.insert(0, 0)
    x_intersections.append(24*num_days)
    if diff[0]>0:
        array1 = [[x_intersections[i], x_intersections[i+1]] for i in range(0, len(x_intersections)-1, 2)]
        array2 = [[x_intersections[i], x_intersections[i+1]] for i in range(1, len(x_intersections)-1, 2)]
    else:
        array2 = [[x_intersections[i], x_intersections[i+1]] for i in range(0, len(x_intersections)-1, 2)]
        array1 = [[x_intersections[i], x_intersections[i+1]] for i in range(1, len(x_intersections)-1, 2)]

    return array1, array2

def hours_to_date(start_datetime,array):
    result=[]
    for iv in array:
        start_hour = int(iv[0])
        start_minute = int((iv[0] - start_hour) * 60)
        end_hour = int(iv[1])
        end_minute = int((iv[1] - end_hour) * 60)
        start_time = start_datetime + timedelta(hours=start_hour, minutes=start_minute)
        end_time = start_datetime + timedelta(hours=end_hour, minutes=end_minute)
        result1=start_time.strftime('%Y-%m-%d %H:%M') + ' - ' + end_time.strftime('%Y-%m-%d %H:%M')
        result.append(result1)
        result_2013 = [s.replace('2019', '2013') for s in result]
    return result_2013

def timestamp15(start,end):
    interval = timedelta(minutes=15)
    timestamps = [start + interval * i for i in range((end - start) // interval + 1)]
    return timestamps

def Windavailabilty(timestamps,non_availability):
    time_intervals = [
        (datetime.strptime(s[:16], '%Y-%m-%d %H:%M'), datetime.strptime(s[19:], '%Y-%m-%d %H:%M'))
        for s in non_availability]
    # Check if each timestamp belongs to any of the time intervals
    availability = []
    for ts in timestamps:
        is_available = 1
        for interval_start, interval_end in time_intervals:
            if interval_start <= ts <= interval_end:
                is_available = 0
                break
        availability.append((ts.strftime('%Y-%m-%d %H:%M'), is_available))
    # Convert the availability list to a Pandas DataFrame
    #availability_df = pd.DataFrame(availability, columns=['time_interval', 'availability'])
    # Set the timestamp as the index
    #availability_df['time_interval'] = pd.to_datetime(availability_df['time_interval'])
    #availability_df.set_index('time_interval', inplace=True)
    return pd.Series(availability)

def scheduling(driving_data,SOCmin,SOCmax,SOCmedian,deltaEnergy,CB,r_c,r_d):
    T_d=[]
    Energy_level = [CB]
    SOC = [1]
    dischargedEnergy=[]
    for t in range(1,len(driving_data)):
        if driving_data.iloc[t] == 0:
            ## driving
            SOCnew = (Energy_level[t - 1]-deltaEnergy) / CB
            Energy_level.append(Energy_level[t - 1] - deltaEnergy)
            SOC.append(SOCnew)
        elif driving_data.iloc[t] == 1 and df_wind_availability.iloc[t][1] == 1:
            # parking and wind positive
            if SOC[t-1]<SOCmax:
                #parking and charging
                SOCnew = (Energy_level[t - 1]+r_c * T_i) / CB
                if SOCnew>=SOCmax:
                    SOC.append(SOCmax)
                    Energy_level.append(CB)
                else:
                    SOC.append(SOCnew)
                    Energy_level.append(Energy_level[t - 1] + r_c * T_i)
            elif SOC[t-1]==SOCmax:
                #parking
                SOC.append(SOCmax)
                Energy_level.append(CB)
        elif driving_data.iloc[t] == 1 and df_wind_availability.iloc[t][1] == 0:
            #parking and wind negative
            if SOC[t-1] <= SOCmin:
                #parking and charging
                SOCnew = (Energy_level[t - 1]+r_c * T_i) / CB
                if SOCnew > SOCmedian:
                    SOC.append(SOCmedian)
                    Energy_level.append(SOCmedian*CB)
                else:
                    SOC.append(SOCnew)
                    Energy_level.append(Energy_level[t - 1] + r_c * T_i)
            if SOC[t-1]> SOCmin:
                # parking and discharging
                SOCnew = (Energy_level[t - 1]-r_d * T_i) / CB
                if SOCnew < SOCmin:
                    SOC.append(SOCmin)
                    Energy_level.append(SOCmin * CB)
                    Tnew=(Energy_level[t - 1]-SOCmin*CB)/(r_d)
                    Energy_dis=Tnew*r_d
                    dischargedEnergy.append(Energy_dis)
                    T_d.append([driving_data.index[t],Tnew])
                else:
                    SOC.append(SOCnew)
                    Energy_level.append(Energy_level[t - 1] - r_d * T_i)
                    T_d.append([driving_data.index[t],T_i])
        else:
            break
        notworking = (SOC[t - 1] < 0) or (Energy_level[t - 1] > CB) or (Energy_level[t - 1] < 0)
        if notworking:
            works=False
            break
        else:
            works=True
    #accSum = list(accumulate(dischargedEnergy))
    return SOC,T_d, works, dischargedEnergy

def optimalParameter(driving_data, SOCmax, deltaEnergy, CB):
    # Define initial and maximum values for a, b, and c
    #SOCmin_initial, SOCmin_max, SOCmin_inc = 0.30, 1.0, 0.05
    #SOCmedian_initial, SOCmedian_max, SOCmedian_inc = 0.65, 1.0, 0.05
    #r_d_initial, r_d_min, r_d_inc = 70, 20, -10
    #r_c_initial, r_c_min, r_c_inc = 50, 150, 20
    #fast
    SOCmin_initial, SOCmin_max, SOCmin_inc = 0.30, 1.0, 0.05
    SOCmedian_initial, SOCmedian_max, SOCmedian_inc = 0.65, 1.0, 0.05
    r_d_initial, r_d_min, r_d_inc = 70, 20, -10
    r_c_initial, r_c_min, r_c_inc = 50, 150, 20
    #b=0.85
    found_optimal = False
    #SOCopt,optimal_output=0
    # Iterate over values of a
    for a in np.arange(SOCmin_initial, SOCmin_max , SOCmin_inc):
        # Iterate over values of b
        a=round(a,2)
        for b in np.arange(SOCmedian_initial, SOCmedian_max, SOCmedian_inc):
            b=round(b,2)
            # Iterate over values of c
            for d in np.arange(r_c_initial, r_c_min + r_c_inc, r_c_inc):
                d=round(d,1)
                # Iterate over values of d
                for c in np.arange(r_d_initial, r_d_min + r_d_inc, r_d_inc):
                    c=round(c,1)
                    SOC,output, works = scheduling(driving_data,a,SOCmax,b,deltaEnergy,CB,d,c)
                    if works:
                        #SOC,output, works = scheduling(medallion,a, SOCmax, b, deltaEnergy, CB, d, c)
                        #count = count + 1
                        #print(count)
                        print(driving_data.name)
                        #print(f"medaillon:{driving_data.name} and SOCmin={a}, SOCmedian={b}, r_c={d},r_d={c}.")
                        found_optimal = True
                        break
                if found_optimal:
                    break
            if found_optimal:
                break
        if found_optimal:
            break
    if not found_optimal:
        a=1
        b=1
        c=0
        d=0
        SOC, output, works = scheduling(driving_data, a, SOCmax, b, deltaEnergy, CB, d, c)
        output=[[0,0]]
        #output = [['2013-01-01 00:00:00', 0]]
        #print(f"medaillon:{driving_data.name} not possible")
    return a,b,c,d,SOC,output
def newoptimalParameter(driving_data, SOCmax, deltaEnergy, CB):
    # Define initial and maximum values for a, b, and c
    #SOCmin_initial, SOCmin_max, SOCmin_inc = 0.30, 1.0, 0.05
    #SOCmedian_initial, SOCmedian_max, SOCmedian_inc = 0.65, 1.0, 0.05
    #r_d_initial, r_d_min, r_d_inc = 70, 20, -10
    #r_c_initial, r_c_min, r_c_inc = 50, 150, 20
    #fast
    SOCmin_initial, SOCmin_max, SOCmin_inc = 0.10, 1.0, 0.05
    SOCmedian_initial, SOCmedian_max, SOCmedian_inc = 0.3, 1.0, 0.05
    r_d_initial, r_d_min, r_d_inc = 70, 20, -10
    r_c_initial, r_c_min, r_c_inc = 50, 100, 20
    #b=0.85
    found_optimal = False
    #SOCopt,optimal_output=0
    # Iterate over values of a
    for b in np.arange(SOCmedian_initial, SOCmedian_max, SOCmedian_inc):
        # Iterate over values of b
        b = round(b, 2)
        for a in np.arange(SOCmin_initial, SOCmin_max, SOCmin_inc):
            a = round(a, 2)
            # Iterate over values of c
            for d in np.arange(r_c_initial, r_c_min + r_c_inc, r_c_inc):
                d = round(d, 1)
                # Iterate over values of d
                for c in np.arange(r_d_initial, r_d_min + r_d_inc, r_d_inc):
                    c = round(c, 1)
                    SOC, output, works,accsum = scheduling(driving_data, a, SOCmax, b, deltaEnergy, CB, d, c)
                    if works:
                        # SOC,output, works = scheduling(medallion,a, SOCmax, b, deltaEnergy, CB, d, c)
                        # count = count + 1
                        # print(count)
                        print(driving_data.name)
                        # print(f"medaillon:{driving_data.name} and SOCmin={a}, SOCmedian={b}, r_c={d},r_d={c}.")
                        found_optimal = True
                        break
                if found_optimal:
                    break
            if found_optimal:
                break
        if found_optimal:
            break
    if not found_optimal:
        a=1
        b=1
        c=0
        d=0
        SOC, output, works = scheduling(driving_data, a, SOCmax, b, deltaEnergy, CB, d, c)
        output=[[0,0]]
        #output = [['2013-01-01 00:00:00', 0]]
        #print(f"medaillon:{driving_data.name} not possible")
    return a,b,c,d,SOC,output,accsum
def createdischargeavail(medallion, T_dopt,a,b,c,d):
    df = pd.DataFrame(T_dopt, columns=['timestamp', medallion])
    df.set_index('timestamp', inplace=True)
    #df = df.groupby('timestamp').sum()
    # Define the desired date range for the index ## CHANGE HERE FOR NEW MONTH
    date_range = pd.date_range(start='2013-01-01 00:00:00', end='2013-01-30 23:45:00', freq='15T')

    # Reindex the DataFrame with the desired date range and fill missing values with 0
    df = df.reindex(date_range, fill_value=0)
    # Custom row labels and values
    row_labels = ['SOCmin', 'SOCmedian', 'r_c', 'r_d']
    values = [a, b, d, c]  # Replace a, b, c, and d with the actual values you want to add

    for label, value in zip(row_labels, values):
        df.loc[label] = value
    return df


##################MAIN WIND #############################
day, time, power_output=interval_dataframe(df, start_time, end_time)
num_days=fnum_days(start_time,end_time)
NYC_demand=NYdemand_days(num_days,NYC_demand)
xnew,NYC_demand_new, power_output_new= interpolation(num_days,NYC_demand,power_output)
wind_minus, wind_plus = intervals(xnew,num_days,NYC_demand_new,power_output_new)
wind_minus= hours_to_date(start_datetime,wind_minus)
wind_plus= hours_to_date(start_datetime,wind_plus)
timestamps=timestamp15(start,end)
df_wind_availability=Windavailabilty(timestamps,wind_minus)

####### TAXI MAIN #######################
#driving_data = df_taxi.loc[:, medallion]
#test2=driving_data.iloc[17]

####### MAIN algorithm ############

# Define a new function that wraps optimalParameter and createdischargeavail

def process_taxi_column(driving_data):
    a, b, c, d, SOCopt, T_dopt,accsum = newoptimalParameter(driving_data, SOCmax, deltaEnergy, CB)
    medaillon_df = createdischargeavail(driving_data.name, T_dopt,a,b,c,d)

    return medaillon_df,accsum

# Get a list of all the medallions from the DataFrame columns


# Create an empty DataFrame to store the results for all medallions
all_medaillon_df = pd.DataFrame()
df_taxi = df_taxi.iloc[:,0:10]
medallion_list = df_taxi.columns
#all_medaillon_df = df_taxi.apply(process_taxi_column, axis=0)
#all_medaillon_df = pd.DataFrame(
    #df_taxi.apply(process_taxi_column, axis=0).tolist(),
    #index=df_taxi.columns
#)
list_of_medaillon_dfs = [process_taxi_column(df_taxi[col])[0] for col in df_taxi.columns]
accsum = [process_taxi_column(df_taxi[col])[1] for col in df_taxi.columns]
print(accsum)
all_medaillon_df = pd.concat(list_of_medaillon_dfs, axis=1)
all_medaillon_df.to_excel("1Privatelist.xlsx", index=False)
# Get the discharging rates from the last row of each column
discharging_rates = all_medaillon_df.iloc[-1]

# Multiply each column by its respective discharging rate
scaled_medaillon_df = all_medaillon_df.iloc[:-4].mul(discharging_rates)
scaled_medaillon_df.to_excel("1Privaterows_scaled.xlsx", index=False)
# Sum all rows except the last 4, since they contain the SOCmin, SOCmedian, r_c, and r_d values
summed_rows = scaled_medaillon_df.iloc[:-4].sum(axis=1)

# Save the summed_rows DataFrame as an Excel file
summed_rows.to_excel("1Private_summed_rows_scaled.xlsx", index=False)
