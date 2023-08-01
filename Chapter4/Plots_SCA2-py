import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


df_1=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month4_rate_c50_rate_d20_WIND9000MW.csv',index_col=0, parse_dates=True)
df_2=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month10_rate_c50_rate_d20_WIND9000MW.csv',index_col=0, parse_dates=True)
df_3=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month11_rate_c50_rate_d20_WIND9000MW.csv',index_col=0, parse_dates=True)
df_4=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month3_rate_c50_rate_d20_WIND9000MW.csv',index_col=0, parse_dates=True)
df_5=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month12_rate_c50_rate_d20_WIND9000MW.csv',index_col=0, parse_dates=True)
df_6=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month6_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_7=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month8_rate_c50_rate_d50_WIND9000MW.csv',index_col=0, parse_dates=True)
df_8=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_1Month1_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_9=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_50Month1_rate_c50_rate_d50_WIND13000MW.csv',index_col=0, parse_dates=True)
df_10=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_100Month1_rate_c50_rate_d50_WIND9000MW.csv',index_col=0, parse_dates=True)
df_11=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_50Month8_rate_c50_rate_d50_WIND9000MW.csv', index_col=0, parse_dates=True)
df_12=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_50Month1_rate_c50_rate_d50_WIND9000MW.csv',index_col=0, parse_dates=True)
df_13=pd.read_csv('/Users/louisahillegaart/pythonProject1/Timestep_scaledown_50Month11_rate_c50_rate_d50_WIND13000MW.csv',index_col=0, parse_dates=True)
df_14=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_5Month1_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_15=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_5Month6_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_16=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_5Month8_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_17=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_5Month11_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_18=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_10Month1_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_19=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_10Month4_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_20=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_10Month11_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)
df_21=pd.read_csv('/Users/louisahillegaart/pythonProject1/2Timestep_scaledown_20Month8_rate_c50_rate_d20_WIND13000MW.csv',index_col=0, parse_dates=True)





df_wind_9 = pd.read_csv(f'Wind_NY_Power9.csv', index_col=0, parse_dates=True)
df_wind_13 = pd.read_csv(f'Wind_NY_Power13.csv', index_col=0, parse_dates=True)
df_wind9_month = df_wind_9[df_wind_9.index.month == 1]
df_wind13_month = df_wind_13[df_wind_13.index.month == 1]

df_wind13_month['new wind']=df_wind13_month['Pdifference']/5 +df_wind13_month['NYC_demand']
# Remove rows with duplicate indices
def removeduplicate(df):
    df = df[~df.index.duplicated(keep='first')]
    # Sort the index in ascending order (datetime)
    df.sort_index(inplace=True)
    return df

df_3=removeduplicate(df_3)
df_4=removeduplicate(df_4)
df_5=removeduplicate(df_5)
df_8=removeduplicate(df_8)
df_9=removeduplicate(df_9)
df_10=removeduplicate(df_10)
df_11=removeduplicate(df_11)
df_12=removeduplicate(df_12)
df_13=removeduplicate(df_13)
df_14=removeduplicate(df_14)
df_15=removeduplicate(df_15)
df_16=removeduplicate(df_16)
df_17=removeduplicate(df_17)
df_18=removeduplicate(df_18)
df_19=removeduplicate(df_19)
df_20=removeduplicate(df_20)
df_21=removeduplicate(df_21)

df_wind9_month['new wind']=df_12['P_difference']+df_wind9_month['NYC_demand']

print(df_1.columns)


def processdata(df):
    df.drop(columns=['free cars'], inplace=True)
    # Check the condition 'P_difference_left' > 0 and update 'Nb' accordingly
    df.loc[df['P_difference_left'] > 0, 'Nb'] = df['parked '] - df['Blocked charging ']

    # Check the condition 'P_difference_left' < 0 and update 'Nb' accordingly
    df.loc[df['P_difference_left'] < 0, 'Nb'] = df['parked '] - df['Blocked discharging ']

    # For P_difference_left > 0
    df.loc[df['P_difference_left'] > 0, 'TotalParticipation needed'] = df['alpha '] * df['Nb'] + df['critical '] + df['driving '] + df['Blocked charging ']

    # For P_difference_left < 0
    df.loc[df['P_difference_left'] < 0, 'TotalParticipation needed'] = df['alpha '] * df['Nb'] + df['critical '] + df['driving '] + df['Blocked discharging ']
    return df


df1= processdata(df_1)
df2= processdata(df_2)
df3= processdata(df_3)
df4= processdata(df_4)
df5= processdata(df_5)
df6= processdata(df_6)
df7= processdata(df_7)
df8= processdata(df_8)
df9= processdata(df_9)
df10= processdata(df_10)
df11=processdata(df_11)
df12=processdata(df_12)
df13=processdata(df_13)
df14=processdata(df_14)
df15=processdata(df_15)
df16=processdata(df_16)
df17=processdata(df_17)
df18=processdata(df_18)
df19=processdata(df_19)
df20=processdata(df_20)
df21=processdata(df_21)



# Assuming df1, df2, and df3 are your dataframes
# First, make sure your indices are DatetimeIndex, if not convert them
df1.index = pd.to_datetime(df1.index)
df2.index = pd.to_datetime(df2.index)
df3.index = pd.to_datetime(df3.index)
df4.index = pd.to_datetime(df4.index)
df5.index = pd.to_datetime(df5.index)
df6.index = pd.to_datetime(df6.index)
df7.index = pd.to_datetime(df7.index)
df8.index = pd.to_datetime(df8.index)
df9.index = pd.to_datetime(df9.index)
df10.index = pd.to_datetime(df10.index)
df11.index = pd.to_datetime(df11.index)
df12.index = pd.to_datetime(df12.index)
df13.index = pd.to_datetime(df13.index)
df14.index = pd.to_datetime(df14.index)
df15.index = pd.to_datetime(df15.index)
df16.index = pd.to_datetime(df16.index)
df17.index = pd.to_datetime(df17.index)
df18.index = pd.to_datetime(df18.index)
df19.index = pd.to_datetime(df19.index)
df20.index = pd.to_datetime(df20.index)
df21.index = pd.to_datetime(df21.index)

# Subtract the minimum date in each dataframe from every date in the dataframe
df1.index = df1.index - df1.index.min()
df2.index = df2.index - df2.index.min()
df3.index = df3.index - df3.index.min()
df4.index = df4.index - df4.index.min()
df5.index = df5.index - df5.index.min()
df6.index = df6.index - df6.index.min()
df7.index = df7.index - df7.index.min()
df8.index = df8.index - df8.index.min()
df9.index = df9.index - df9.index.min()
df10.index = df10.index - df10.index.min()
df11.index = df11.index - df11.index.min()
df12.index = df12.index - df12.index.min()
df13.index = df13.index - df13.index.min()
df14.index = df14.index - df14.index.min()
df15.index = df15.index - df15.index.min()
df16.index = df16.index - df16.index.min()
df17.index = df17.index - df17.index.min()
df18.index = df18.index - df18.index.min()
df19.index = df19.index - df19.index.min()
df20.index = df20.index - df20.index.min()
df21.index = df21.index - df21.index.min()

# Convert timedelta index into hours
df1.index = df1.index.total_seconds() / 3600
df2.index = df2.index.total_seconds() / 3600
df3.index = df3.index.total_seconds() / 3600
df4.index = df4.index.total_seconds() / 3600
df5.index = df5.index.total_seconds() / 3600
df6.index = df6.index.total_seconds() / 3600
df7.index = df7.index.total_seconds() / 3600
df8.index = df8.index.total_seconds() / 3600
df9.index = df9.index.total_seconds() / 3600
df10.index = df10.index.total_seconds() / 3600
df11.index = df11.index.total_seconds() / 3600
df12.index = df12.index.total_seconds() / 3600
df13.index = df13.index.total_seconds() / 3600
df14.index = df14.index.total_seconds() / 3600*2
df15.index = df15.index.total_seconds() / 3600*2
df16.index = df16.index.total_seconds() / 3600*2
df17.index = df17.index.total_seconds() / 3600*2
df18.index = df18.index.total_seconds() / 3600*2
df19.index = df19.index.total_seconds() / 3600*2
df20.index = df20.index.total_seconds() / 3600*2
df21.index = df21.index.total_seconds() / 3600*2


# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data
ax.plot(df1['TotalParticipation needed'],color='black', label='April (+227.27 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df2['TotalParticipation needed'],color='blue', label='October (+149.93 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df3['TotalParticipation needed'],color='green', label='November(+39.13 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df6['TotalParticipation needed'],color='red', label='June (+58.9 Gwh), C=13000 MW, charging rate=50 Kw, discharging rate=20Kw')

# Set plot title and labels
ax.set_title('Total EV participation (ideal cases : with small positive power balance offsets)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()


# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data
ax.plot(df4['TotalParticipation needed'],color='black', label='March (-22.68 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df5['TotalParticipation needed'],color='blue', label='December (-100.06 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')

# Set plot title and labels
ax.set_title('Total EV participation (with small negative power balance offsets)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()


# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data
ax.plot(df7['TotalParticipation needed'],color='black', label='August (-3432.89 GWh), C=9000 MW, charging rate=50 Kw, discharging rate=50Kw')
ax.scatter(df7.index, df7['TotalParticipation needed'],color='black',s=5)


ax.plot(df8['TotalParticipation needed'],color='blue', label='January (+1850.55 Gwh), C=13000 MW, charging rate=50 Kw, discharging rate=20Kw')

# Set plot title and labels
ax.set_title('Total EV participation (critical cases with high power balance offsets)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()


# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data
ax.plot(df9['TotalParticipation needed'],color='black', label='January, C=13000 MW, eta =50,  charging rate=50 Kw, discharging rate=50Kw')
#ax.scatter(df9.index, df9['TotalParticipation needed'],color='black',s=5)

ax.plot(df10['TotalParticipation needed'],color='blue', label='January, C=9000 MW, eta =100,  charging rate=50 Kw, discharging rate=50Kw')
ax.plot(df11['TotalParticipation needed'],color='green', label='August , C=9000 MW, eta =50,  charging rate=50 Kw, discharging rate=50Kw')
ax.plot(df12['TotalParticipation needed'],color='red', label='January , C=9000 MW, eta =50,  charging rate=50 Kw, discharging rate=50Kw')
ax.plot(df13['TotalParticipation needed'],color='pink', label='November , C=13000 MW, eta =50,  charging rate=50 Kw, discharging rate=50Kw')

# Set plot title and labels
ax.set_title('Total EV participation (critical cases with high power balance offsets)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()



# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data
ax.plot(df7['Nb']*df7['alpha '],color='black', label='August (-3432.89 GWh), C=9000 MW, charging rate=50 Kw, discharging rate=50Kw')
#ax.scatter(df7.index, df7['TotalParticipation needed'],color='black',s=5)
ax.plot(df8['Nb']*df8['alpha '],color='blue', label='January (+1850.55 Gwh), C=13000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df3['Nb']*df3['alpha '],color='green', label='November(+39.13 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df2['Nb']*df2['alpha '],color='blue', label='October (+149.93 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')

ax.plot(df5['Blocked discharging ']+df5['Blocked charging '],color='red', label='December (-100.06 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df4['Blocked discharging ']+df4['Blocked charging '],color='orange', label='March (-22.68 Gwh), C=9000 MW, charging rate=50 Kw, discharging rate=20Kw')

# Set plot title and labels⁄
ax.set_title('Total EV participation (Blocked)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()


# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data

ax.plot(df14['TotalParticipation needed'],color='blue', label='January, C=13000 MW, eta =5,  charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df15['TotalParticipation needed'],color='green', label='June , C=13000 MW, eta =5,  charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df16['TotalParticipation needed'],color='red', label='August , C=13000 MW, eta =5,  charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df17['TotalParticipation needed'],color='pink', label='November , C=13000 MW, eta =5,  charging rate=50 Kw, discharging rate=20Kw')

# Set plot title and labels
ax.set_title('Total EV participation (with eta :5)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()



# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot your data

ax.plot(df18['TotalParticipation needed'],color='blue', label='January, C=13000 MW, eta =10,  charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df19['TotalParticipation needed'],color='green', label='June , C=13000 MW, eta =10,  charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df21['TotalParticipation needed'],color='red', label='August , C=13000 MW, eta =10,  charging rate=50 Kw, discharging rate=20Kw')
ax.plot(df20['TotalParticipation needed'],color='pink', label='November , C=13000 MW, eta =10,  charging rate=50 Kw, discharging rate=20Kw')

# Set plot title and labels
ax.set_title('Total EV participation (with eta :10)', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12) # adjust the label here
ax.set_ylabel('Total EV participation', fontsize=12)

# Display the legend
ax.legend()

# Set grid
ax.grid(True)

# Show the plot
plt.show()
