import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from scipy.integrate import simps, quad
from datetime import datetime
statementplot=True
statementhours=True

capacityUS=9000
#date='2019-01-02'

#df = pd.read_excel('Timeseries60min.xlsx', parse_dates=['Datetime'], index_col='Datetime')
## with data from the US
df = pd.read_csv('USWINDDATA_process.csv',parse_dates=['Local_time_New_York'], index_col='Local_time_New_York', delimiter=';')
start_time = '2019-01-01 00:00:00'
end_time = '2019-01-07 23:00:00'
day = df.loc[start_time:end_time]

#day = df.loc[date] this is only if you want to plot a day or a ful month
time = day.index
#power_output = day['Wind_generation_MW'] # indes is the date and time and the values are the power output
# US winddata
power_output=day['Electricity'].to_numpy()/1000
###
#capacity= day['Wind_Capacity'][1]
#num_days=1
def fnum_days(start_time,end_time):
    start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    time_diff = end_datetime - start_datetime
    num_days = time_diff.days + 1  # Add one day to include January 1st
    return num_days
num_days=fnum_days(start_time,end_time)
x=np.arange(0,24*num_days,1)
NYC_demand=[4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801]

NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])#print('Original capacity is: ',capacity)
#x=np.arange(0,Number,1)

# Repeat the array for every day#
NYC_demand = np.tile(NYC_demand, num_days)
#scaling_number= capacity/power_output
#power_ouput_scaled=capacityUS/scaling_number
#print(power_output)
#print('Original capacity is: ',capacity)
#print(power_ouput_scaled)

#xx=np.linspace(0,24,300)
#######
f = interpolate.interp1d(x, NYC_demand, kind='cubic')
#g=interpolate.interp1d(x, power_ouput_scaled, kind='cubic')
#USwinddata
g=interpolate.interp1d(x, power_output, kind='cubic')
# Create a new x array with higher resolution
xnew = np.linspace(0, 23*num_days, 1000)
# Evaluate the interpolated function at the new x values
NYC_demand_new = f(xnew)
power_output_new=g(xnew)

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
if statementhours:
    print("Hourly Intervals where Demand is higher:", array1)
    print("Hourly Intervals where Wind generation is higher:", array2)

def Totalhours(array):
# Convert the list to a numpy array
    my_array = np.array(array)
    if len(array)>=1:
        # Compute the pairwise differences between the values in each list of each element
        diffs = np.abs(np.diff(my_array, axis=1))
        # Sum up the differences over the entire array
        total_diff = np.sum(diffs)
        return total_diff
    else:
        return 0
TotalHours_NotAvailable= Totalhours(array1)
TotalHours_Available= Totalhours(array2)
if statementhours:
    print("Total Hours where Demand is higher:",TotalHours_NotAvailable)
    print("Total Hours where Wind Power is higher:",TotalHours_Available)



# Plot the interpolated curve
def plotting(statement):
    if statement:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(xnew, power_output_new, label='Wind Offshore Generation')
        ax.plot(xnew, NYC_demand_new, label='NYC energy demand')
        plt.fill_between(xnew, NYC_demand_new, power_output_new, where=NYC_demand_new >= power_output_new, interpolate=True, alpha=0.5, color='grey')
        plt.fill_between(xnew, NYC_demand_new, power_output_new, where=NYC_demand_new < power_output_new, interpolate=True, alpha=0.5, color='green')
        # Set the plot title and axis labels
        ax.set(title=f'Wind Offshore Generation vs. Demand Forecast ({start_time} until {end_time})', xlabel='Time (hours)', ylabel='Power Output (MW)')

        # Customize the x-axis ticks
        #plt.xticks(np.arange(0, 24*num_days, 1))
        # Add a legend to the plot
        ax.legend()
        plt.show()
    else:
        return 0

plotting(statementplot)



mask = power_output_new >= NYC_demand_new
if mask.all():
    area = simps(power_output_new, xnew)-simps(NYC_demand_new, xnew) 
    print("Energy generated is covering all of the demand of NY over the whole day. The energy available is (Mwh):", area/24)
elif (~mask).all():
    area = simps(NYC_demand_new, xnew) - simps(power_output_new, xnew)
    print("Energy demanded is always higher than Energy generate by wind. The energy not available is (Kwh):", area/24)
else:
    mask = NYC_demand_new > power_output_new
    print(len(NYC_demand_new))
    #print(mask)
    x1 = xnew[mask]
    #print(x1)
    y1 = NYC_demand_new[mask]
    #print(y1)
    y2 = power_output_new[mask]
    mask2 = y1>y2
    if mask2.all():
        print('correct')
    #print(y2)
    area1 = simps(y1-y2, x1)
    ar1=simps(y1,x1)
    ar2 = simps(y2, x1)
    #print(area1)

    # calculate area where yNY is less than yWIND
    x2 = xnew[~mask]
    print(len(x2))
    y3 = NYC_demand_new[~mask]
    y4 = power_output_new[~mask]
    area2 = simps(y4 - y3, x2)
    Energyforstorage=(area2-area1)/24
    # calculate intervals of aval.
    x1_interval = (min(x1), max(x1))
    x2_interval = (min(x2), max(x2))

    # print the two areas
    print("Energy not available:", area1/TotalHours_NotAvailable)
    print("Excess Energy available:", area2/TotalHours_Available)
    print("Diff in Energy:", Energyforstorage)
    #print ("Timeframe of non availibility:", x1_interval)
    #print ("Timeframe of availibility:", x2_interval)
    prob=[]
    if Energyforstorage>0:
        prob.append(1)
        print('The wind energy covers the demand of NY')
        print('Min. Energy that needs to be stored in EVs per day is :', area1/TotalHours_NotAvailable)
        print('Max. Energy that could be stored in EVs per day is :', Energyforstorage)
    else:
        print('The wind energy does not cover the demand of NY')
    print(prob)