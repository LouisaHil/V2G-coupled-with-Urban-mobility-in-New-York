import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from scipy.integrate import simps, quad
from datetime import datetime, timedelta
statementplot=True
statementhours=True

capacityUS=9000
date='2017-01' ## for march and october not working
df = pd.read_excel('Timeseries60min.xlsx', parse_dates=['Datetime'], index_col='Datetime')
day = df.loc[date]
print(day)
time = day.index
power_output = day['Wind_generation_MW'] # index is the date and time and the values are the power output
capacity= day['Wind_Capacity'][1]
print('OG capacity of wind farm ',capacity)


start_date = datetime(2017, 1, 1)  # Start date
end_date = datetime(2017, 1, 31)  # End date
scaling_number= capacity/power_output
power_output_scaled=capacityUS/scaling_number
Number=len(power_output_scaled)
print(Number)
## Compute the mean of the non-missing values in power_output_scaled
mean_power = np.nanmean(power_output_scaled)
## Fill the missing values in power_output_scaled with the mean value
power_output_scaled[np.isnan(power_output_scaled)] = mean_power
num_days = (end_date - start_date).days +1  # Number of days between start and end date#power_ouput_scaled=capacityUS/scaling_number
print(num_days)
##print(power_output)
NYC_demand = np.array([4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801])#print('Original capacity is: ',capacity)
x=np.arange(0,Number,1)

# Repeat the array for every day#
NYC_demand_per_day = np.tile(NYC_demand, num_days)

########  load nb of evs #####
# Load the data from a CSV file
date2='2013-01'
data = pd.read_csv('merged.csv')
data_filtered = data[data['time_interval'].str.startswith('2013-01')]
scale=len(data_filtered)
x2=np.arange(0,scale,1)
NbofEvs=data_filtered['total']
batteryCapacity=NbofEvs*70/1000
########################
f = interpolate.interp1d(x, NYC_demand_per_day, kind='cubic')
g=interpolate.interp1d(x, power_output_scaled, kind='cubic')
h=interpolate.interp1d(x2, NbofEvs, kind='cubic')
e=interpolate.interp1d(x2, batteryCapacity, kind='cubic')
## Create a new x array with higher resolution
xnew = np.linspace(0, Number-1, 87590)
## Evaluate the interpolated function at the new x values
NYC_demand_new = f(xnew)
power_output_new=g(xnew)
NbofEvs_new=h(xnew)
batteryCapacity_new=e(xnew)
#
### define the intersection points :
diff = NYC_demand_new - power_output_new
sign_changes = np.where(np.diff(np.sign(diff)))[0]
x_intersections = xnew[sign_changes]
x_intersections = x_intersections.tolist()  # Convert to list
print(x_intersections)
x_intersections.insert(0, 0)
x_intersections.append(8784)
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
       # fig, ax = plt.subplots(figsize=(12, 6))
        fig, ax1 = plt.subplots(figsize=(16, 10))
        ax2 = ax1.twinx()
        ax1.plot(xnew/4, power_output_new, label='Wind Offshore Generation')
        ax1.plot(xnew/4, NYC_demand_new, label='NYC energy demand')
        ax2.plot(xnew/4,batteryCapacity_new,label='Total Battery capacity MWh',color='red')
        plt.fill_between(xnew/4, NYC_demand_new, power_output_new, where=NYC_demand_new >= power_output_new, interpolate=True, alpha=0.5, color='grey')
        plt.fill_between(xnew/4, NYC_demand_new, power_output_new, where=NYC_demand_new < power_output_new, interpolate=True, alpha=0.5, color='green')
        # Set the plot title and axis labels
        ax2.set_ylabel('Total Battery capacity MWh')

    # Add a legend to the plot
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    # only show half of the x-axis
        #ax1.set_xlim([0, scale/4])
        #ax2.set_xlim([0, scale/4])

        # Adjust the aspect ratio of the plot
        #ax1.set_aspect((scale/2) / ax1.get_ylim()[1])
        #fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

        ax1.set(title=f'Wind Offshore Generation vs. Demand Forecast ({date})', xlabel='Time', ylabel='Power Output (MW)')

        # Customize the x-axis ticks
        #plt.xticks(np.arange(0, 24, 1))
        # Add a legend to the plot
        #ax1.legend()
        plt.show()
    else:
        return 0

plotting(statementplot)


# Calculate the area under the curve using the trapezoidal rule
area1 = np.trapz(NYC_demand_new, xnew)
area2 = np.trapz(power_output_new, xnew)
print("Area under the curve NY demand :", area1/1000)
print("Area under the curve Wind power :", area2/1000)
print("Diff Area under the curve Wind-NyDemand :", (area2-area1)/1000)












#mask = power_output_new >= NYC_demand_new
#if mask.all():
#    area = simps(power_output_new, xnew)-simps(NYC_demand_new, xnew)
#    print("Energy generated is covering all of the demand of NY over the whole day. The energy available is (Mwh):", area/24)
#elif (~mask).all():
#    area = simps(NYC_demand_new, xnew) - simps(power_output_new, xnew)
#    print("Energy demanded is always higher than Energy generate by wind. The energy not available is (Kwh):", area/24)
#else:
#    mask = NYC_demand_new > power_output_new
#    print(len(NYC_demand_new))
#    #print(mask)
#    x1 = xnew[mask]
#    #print(x1)
#    y1 = NYC_demand_new[mask]
#    #print(y1)
#    y2 = power_output_new[mask]
#    mask2 = y1>y2
#    if mask2.all():
#        print('correct')
#    #print(y2)
#    area1 = simps(y1-y2, x1)
#    ar1=simps(y1,x1)
#    ar2 = simps(y2, x1)
#    #print(area1)
#
#    # calculate area where yNY is less than yWIND
#    x2 = xnew[~mask]
#    print(len(x2))
#    y3 = NYC_demand_new[~mask]
#    y4 = power_output_new[~mask]
#    area2 = simps(y4 - y3, x2)
#    Energyforstorage=(area2-area1)/24
#    # calculate intervals of aval.
#    x1_interval = (min(x1), max(x1))
#    x2_interval = (min(x2), max(x2))
#
#    # print the two areas
#    print("Energy not available:", area1/TotalHours_NotAvailable)
#    print("Excess Energy available:", area2/TotalHours_Available)
#    print("Diff in Energy:", Energyforstorage)
#    #print ("Timeframe of non availibility:", x1_interval)
#    #print ("Timeframe of availibility:", x2_interval)
#    prob=[]
#    if Energyforstorage>0:
#        prob.append(1)
#        print('The wind energy covers the demand of NY')
#        print('Min. Energy that needs to be stored in EVs per day is :', area1/TotalHours_NotAvailable)
#        print('Max. Energy that could be stored in EVs per day is :', Energyforstorage)
#    else:
#        print('The wind energy does not cover the demand of NY')
#    print(prob)