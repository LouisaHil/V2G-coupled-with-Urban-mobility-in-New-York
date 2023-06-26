import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
#df3 = pd.read_csv('df_Power_sum11.csv')
#df2=pd.read_csv('df_SOC11.csv')
df=pd.read_csv('df_Power_month8csv')
df4=pd.read_csv('df_SOC_month8.csv')

# Extract the first column as 'time_interval'
df['time_interval'] = pd.to_datetime(df.iloc[:, 0])
df4['time_interval'] = pd.to_datetime(df4.iloc[:, 0])
# Filter data for January only
df_january = df[(df['time_interval'].dt.month == 8)]
#df_january = df
df4_january=df4[(df4['time_interval'].dt.month == 8)]

# Set the index as 'time_interval'
df_january.set_index('time_interval', inplace=True)
df4.set_index('time_interval', inplace=True)

# Create the 'power_output + Sum' column
df_january['power_output + Sum'] = df_january['power_output'] + df_january['Sum']
df_january = df_january.iloc[1:]
df_january['Totalcars']=df_january['Charging']+df_january['Discharging']+df_january['Critical']
max_cars = df_january['Totalcars'].max()

# Create the 'power_output + Sum' column
# Set the index as 'time_interval'
# Extract the first column as 'time_interval'
df3['time_interval'] = pd.to_datetime(df3.iloc[:, 0])
df3.set_index('time_interval', inplace=True)
df3['power_output + Sum'] = df3['power_output'] + df3['Sum']
df3 = df3.iloc[1:]
df3['Totalcars']=df3['Charging']+df3['Discharging']+df3['Critical']
max_cars3 = df3['Totalcars'].max()
# Plot the data
plt.figure(figsize=(14, 10))
#plt.plot(df_january.index, df_january['power_output'], color='blue', label='power_output')
plt.plot(df_january.index, df_january['NYC_demand'], color='red', label='NYC_demand')
plt.plot(df_january.index, df_january['power_output + Sum'], color='black', label='power_output + Sum')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Data for January')
plt.legend()
plt.xticks(rotation=45)
plt.show()
# Plot the data
#
#plt.figure(figsize=(14, 10))
##plt.plot(df_january.index, df_january['power_output'], color='blue', label='power_output')
##plt.plot(df_january.index, df_january['NYC_demand'], color='red', label='NYC_demand')
#plt.plot(df_january.index, df_january['power_output + Sum'], color='black', label='power_output + Sum')
#plt.plot(df3.index, df3['power_output + Sum'], color='red', label='power_output + Sum')
#
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.title('Data for January')
#plt.legend()
#plt.xticks(rotation=45)
#plt.show()
#
#plt.figure(figsize=(14, 10))
##plt.plot(df_january.index, df_january['power_output'], color='blue', label='power_output')
##plt.plot(df_january.index, df_january['NYC_demand'], color='red', label='NYC_demand')
#plt.plot(df3.index, df3['power_output + Sum'], color='red', label='power_output + Sum')
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.title('Data for January')
#plt.legend()
#plt.xticks(rotation=45)
#plt.show()
#
## Plot the data
#plt.figure(figsize=(16, 12))
#plt.plot(df_january.index, df_january['Charging']+df_january['Critical'], color='green', label='Charging')
##plt.plot(df3.index, df3['Charging']+df3['Critical'], color='black', label='Charging')
#
#plt.plot(df_january.index, df_january['Discharging'], color='red', label='Discharging')
##plt.plot(df3.index, df3['Discharging'], color='blue', label='Discharging')
#
##plt.plot(df_january.index, df_january['Driving'], color='blue', label='Driving')
##plt.plot(df3.index, df3['Driving'], color='green', label='Driving')
#
#plt.plot(df_january.index, df_january['Totalcars'], color='grey', label=f'Total Cars (Max: {max_cars}')
##plt.plot(df3.index, df3['Totalcars'], color='red', label=f'Total Cars (Max: {max_cars3}')
#
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.title('Column Driving, Charging and Discharging Data for January')
#plt.legend()
#plt.xticks(rotation=45)
#plt.show()
#
## Plot the data
#plt.figure(figsize=(16, 12))
#plt.plot(df2.index, df2['3000500'], color='green', label='medaillon1')
#plt.plot(df4_january.index, df4_january['3000500'], color='blue', label='medaillon1')
#
##plt.plot(df2.index, df2['3000689'], color='red', label='medaillon2')
##plt.plot(df2.index, df2['3000750'], color='blue', label='medaillon3')
##plt.plot(df_january.index, df_january['Charging']+df_january['Critical']+df_january['Driving']+df_january['Discharging'], color='grey', label=f'Total Cars (Max: {max_cars}')
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.title('SOC random car')
#plt.legend()
#plt.xticks(rotation=45)
#plt.show()