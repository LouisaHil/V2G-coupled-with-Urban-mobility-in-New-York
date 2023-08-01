import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta




# Step 1: Read the CSV file and set the datetime index
#df = pd.read_csv('/Users/louisahillegaart/pythonProject1/RESULTS/scenario1_3/SOC_month4_W9_alpha_300_SOC_discharging_19.csv', index_col=0, parse_dates=True)
#df=pd.read_csv('/Users/louisahillegaart/pythonProject1/RESULTS/scenario2_1/BIG5_SOC_month4_W9_alpha_5000_SOC_discharging_20.csv',index_col=0, parse_dates=True)
#df=pd.read_csv('/Users/louisahillegaart/pythonProject1/RESULTS/scenario1_2/BIG5_SOC_month4_W9_alpha_4000_SOC_discharging_11.csv',index_col=0, parse_dates=True)
#df=pd.read_csv('/Users/louisahillegaart/pythonProject1/RESULTS/scenario1_2/BIG5_SOC_month3_W9_alpha_4000_SOC_discharging_11.csv',index_col=0, parse_dates=True)
#df=pd.read_csv('/Users/louisahillegaart/pythonProject1/NEW_10_SOC_month1_W9_alpha_100_SOC_discharging_20.csv',index_col=0, parse_dates=True)
#df_power=pd.read_csv('/Users/louisahillegaart/pythonProject1/NEW_10Power_month1_W9alpha_100_discharging_20.csv',index_col=0, parse_dates=True)
df=pd.read_csv('/Users/louisahillegaart/pythonProject1/NEW_10_SOC_month1_W9_alpha_500_SOC_discharging_20.csv',index_col=0,parse_dates=True)
#print(df_power.columns)
df_power=pd.read_csv('/Users/louisahillegaart/pythonProject1/NEW_10Power_month1_W9alpha_500_discharging_20.csv', index_col=0, parse_dates=True)
#df_power=pd.read_csv('/Users/louisahillegaart/pythonProject1/RESULTS/scenario2_1/BIG_5Power_month4_W9alpha_5000_discharging_20.csv',index_col=0, parse_dates=True)
# Step 2: Calculate the mean SOC values per row
df['Mean_SOC'] = df.mean(axis=1)

# Step 3: Plot the mean SOC values over time
plt.figure(figsize=(10, 6))  # Set the size of the plot (adjust as needed)
plt.plot(df.index, df['Mean_SOC'], color='blue', label='EV fleet= 264940 and alpha=500')

plt.xlabel('Date')
plt.ylabel('Mean SOC Value')
plt.title('Mean SOC Values Over Time in January')
plt.legend()
plt.grid(True)

# Rotate the x-axis tick labels to 45 degrees
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

driving=df_power['P_driving']*df_power['Total']
max_missing_p=max(df_power['Missing_noscale'])
blocked=df_power['P_Blocked_low']*df_power['Total']+df_power['P_Blocked_high']*df_power['Total']
total=df_power['P_Blocked_low']*df_power['Total']+df_power['Charging']+df_power['Discharging']+driving+ df_power['P_Blocked_high']*df_power['Total']

# Calculate the logarithm of the 'Total_needed' column
#log_values = np.log(df_power['Total_needed'])
plt.figure(figsize=(10, 6))  # Set the size of the plot (adjust as needed)
#plt.plot(df_power.index, df_power['Total'], color='black', label='EV fleet= 26940 and alpha=100')
plt.plot(df_power.index, total, color='black', label='EV fleet= 26940 and alpha=500')
#plt.plot(df_power.index, log_values, color='black', label='Logarithm of Total Evs with EV fleet= 13470 and alpha=5000')
plt.xlabel('Date')
plt.ylabel('Total needed (logarithmic)')
plt.title('Total EVs needed')
plt.legend()
plt.grid(True)

# Rotate the x-axis tick labels to 45 degrees
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))  # Set the size of the plot (adjust as needed)
#plt.plot(df_power.index, df_power['Total_needed'], color='black', label='EV fleet= 13470 and alpha=5000')
#plt.plot(df_power.index, df_power['P_Blocked_high'], color='blue', label='P blocked')
plt.plot(df_power.index, df_power['P_Blocked_high']+df_power['P_Blocked_low']+ df_power['P_driving'], color='black', label='Blocked and driving')
#plt.plot(df_power.index, df_power['P_driving'], color='green', label='Driving')

plt.xlabel('Date')
plt.ylabel('Probability')
plt.title('Probability of blocked and driving states over time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))  # Set the size of the plot (adjust as needed)
#plt.plot(df_power.index, df_power['Total_needed'], color='black', label='EV fleet= 13470 and alpha=5000')
plt.plot(df_power.index, -df_power['Pdifference'], color='blue', label='|PWind-PNY|')
plt.plot(df_power.index, -df_power['Pdifference_left'], color='red', label='|PWind-PNY+Pcritical|')
plt.plot(df_power.index, df_power['PV2G'], color='green', label='PV2G')

plt.xlabel('Date')
plt.ylabel('Power MW')
plt.title('Power difference(s) to sustain power balance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

driving=df_power['P_driving']*df_power['Total']
plt.figure(figsize=(10, 6))  # Set the size of the plot (adjust as needed)
#plt.plot(df_power.index, df_power['Total_needed'], color='black', label='EV fleet= 13470 and alpha=5000')
plt.plot(df_power.index, df_power['Missing_noscale'], color='blue', label=f'Missing max: {max_missing_p} Evs')
plt.plot(df_power.index, df_power['Charging'], color='red', label='Charging')
plt.plot(df_power.index, df_power['Critical'], color='green', label='Critical')
plt.plot(df_power.index, df_power['Discharging'], color='grey', label='Discharging')
plt.plot(df_power.index, driving, color='orange', label='Driving')
plt.plot(df_power.index, blocked, color='purple', label='Blocked')
#plt.plot(df_power.index, total, color='black', label='Total EVs')
#plt.plot(df_power.index, df_power['P_Blocked_low']*df_power['Total'], color='pink', label='Blocked_low')
plt.xlabel('Date')
plt.ylabel('Number of EVs')
plt.title('Number of EVs in different states + missing EVS to reach power balance for N=26940 and alpha=500')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()