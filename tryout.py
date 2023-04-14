import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from scipy.integrate import simps

capacityUS=9000
statementplot=True
statementhours=True
start_date = '2015-01-02'
end_date = '2015-12-31'
x = np.arange(0, 24, 1)
NYC_demand = [4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801]
prob=[]
counter=0
# create an empty dataframe where you store the results
results_df = pd.DataFrame(columns=['Counter','Date','Hours Demand','Hours Wind','Total Hours Not Available','Total Hours Available','Energy available','Energy not available','Demand covered 0/1','Min Energy stored', 'Max Energy stored'])

#plotting=False
Energy_not_avail=[]
Energy_avail=[]
Diff_in_Energy=[]
MinEnergy_stor_pd=[]
MaxEnergy_stor_pd=[]
HoursAvail=[]
HoursNotAvail=[]
for date in pd.date_range(start=start_date, end=end_date, freq='D'):
    date_str = date.strftime('%Y-%m-%d')
    counter=counter+1
    print(counter)
    print(date_str)
    df = pd.read_excel('Timeseries60min.xlsx', parse_dates=['Datetime'], index_col='Datetime')
    day = df.loc[date_str]
    power_output = day['Wind_generation_MW']
    capacity = day['Wind_Capacity'][1]
    scaling_number=capacity/power_output
    power_output_scaled=capacityUS/scaling_number

    x=np.arange(0,24,1)
    NYC_demand=[4642,4478,4383,4343,4387,4633,5106,5548,5806,5932,5971,5965,5983,5979,5956,5974,6031,6083,5971,5818,5646,5418,5122,4801]
    scaling_number= capacity/power_output
    power_output_scaled=capacityUS/scaling_number
    #print(power_output)
    print('Original capacity is: ',capacity)
    #print(power_ouput_scaled)

    #######
    f = interpolate.interp1d(x, NYC_demand, kind='cubic')
    g=interpolate.interp1d(x, power_output_scaled, kind='cubic')
    # Create a new x array with higher resolution
    xnew = np.linspace(0, 23, 1000)
    # Evaluate the interpolated function at the new x values
    NYC_demand_new = f(xnew)
    power_output_new=g(xnew)

    ## define the intersection points :
    diff = NYC_demand_new - power_output_new
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    x_intersections = xnew[sign_changes]
    x_intersections = x_intersections.tolist()  # Convert to list
    x_intersections.insert(0, 0)
    x_intersections.append(24)
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
    HoursNotAvail.append(TotalHours_NotAvailable)
    HoursAvail.append(TotalHours_Available)
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
            ax.set(title=f'Wind Offshore Generation vs. Demand Forecast ({date})', xlabel='Time (hours)', ylabel='Power Output (MW)')

            # Customize the x-axis ticks
            plt.xticks(np.arange(0, 24, 1))
            # Add a legend to the plot
            ax.legend()
            plt.show()
        else:
            return 0

    plotting(statementplot)
    mask = power_output_new >= NYC_demand_new

    if mask.all():
        area = simps(power_output_new, xnew)-simps(NYC_demand_new, xnew)
        print("Energy generated is always higher than the demand of NY over the whole day. The energy available is (Mwh):", area/24)
    elif (~mask).all():
        area = simps(NYC_demand_new, xnew) - simps(power_output_new, xnew)
        print("Energy demanded is always higher than Energy generate by wind. The energy not available is (Kwh):", area/24)
    else:
        mask = NYC_demand_new >= power_output_new
        x1 = xnew[mask]
        y1 = NYC_demand_new[mask]
        y2 = power_output_new[mask]
        area1 = simps(y1 - y2, x1)
        # calculate area where yNY is less than yWIND
        x2 = xnew[~mask]
        y3 = NYC_demand_new[~mask]
        y4 = power_output_new[~mask]
        area2 = simps(y4 - y3, x2)
        Energyforstorage=(area2-area1)/24
        # calculate intervals of aval.
        x1_interval = (min(x1), max(x1))
        x2_interval = (min(x2), max(x2))

        # print the two areas
        print("Energy not available:", area1/TotalHours_NotAvailable)
        Energy_not_avail.append(area1/TotalHours_NotAvailable)
        print("Energy available:", area2/TotalHours_Available)
        Energy_avail.append(area2/TotalHours_Available)
        print("Diff in Energy:", Energyforstorage)
        Diff_in_Energy.append(Energyforstorage)
        #print ("Timeframe of non availibility:", x1_interval)
        #print ("Timeframe of availibility:", x2_interval)

        if Energyforstorage>0:
            prob.append(1)
            print(prob)
            print('The wind energy covers the demand of NY')
            print('Min. Energy that needs to be stored in EVs per day is :', area1/TotalHours_NotAvailable)
            MinEnergy_stor_pd.append(area1/TotalHours_NotAvailable)
            print('Max. Energy that could be stored in EVs per day is :', Energyforstorage)
            MaxEnergy_stor_pd.append(Energyforstorage)
        else:
            print('The wind energy does not cover the demand of NY')
            MinEnergy_stor_pd.append(0)
            MaxEnergy_stor_pd.append(0)
print(prob)
probability=sum(prob)/364
print(probability)
print(Diff_in_Energy)
print(Energy_avail)
print(Energy_not_avail)
print(MinEnergy_stor_pd)
print(MaxEnergy_stor_pd)