# Assuming you already have df15 as the DataFrame loaded from the CSV file
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from scipy.integrate import simps, quad
from datetime import datetime
statementplot=True
statementhours=True
capacity=13
df15 = pd.read_csv(f'Wind_NY_Power{capacity}.csv', index_col=0, parse_dates=True)
df_wind13=pd.read_csv(f'Wind_NY_Power13.csv', index_col=0, parse_dates=True)



# Filter data for January
df_january = df15[df15.index.month == 1]  # Replace "1" with the desired month number (e.g., 1 for January)

# Calculate the area between the curves (excess energy and deficient energy)
df_january['area_excess_energy'] = (df_january['power_output'] - df_january['NYC_demand']).clip(0) * 0.25
df_january['area_deficient_energy'] = (df_january['NYC_demand'] - df_january['power_output']).clip(0) * 0.25

# Plotting the power curves and filled areas
plt.figure(figsize=(10, 6))
plt.plot(df_january.index, df_january['power_output'], label='Power Output (MW)')
plt.plot(df_january.index, df_january['NYC_demand'], label='NYC Demand (MW)')

# Filling the areas based on the conditions
plt.fill_between(df_january.index, df_january['power_output'], df_january['NYC_demand'], where=(df_january['power_output'] > df_january['NYC_demand']), interpolate=True, color='green', alpha=0.3)
plt.fill_between(df_january.index, df_january['power_output'], df_january['NYC_demand'], where=(df_january['power_output'] < df_january['NYC_demand']), interpolate=True, color='grey', alpha=0.3)

# Adding labels and legend with energy values
plt.xlabel('Date')
plt.ylabel('Power (MW)')
plt.title(f'Power Output ({capacity}000MW) vs. NYC Demand for AUgust')

# Calculate and display the total excess and deficient energy for January
total_excess_energy = (df_january['area_excess_energy'].sum())/1000
total_deficient_energy = (df_january['area_deficient_energy'].sum())/1000
plt.legend(title=f"Energy (GWh)\nExcess: {total_excess_energy:.2f} GWh\nDeficient: {total_deficient_energy:.2f} GWh")
print(total_excess_energy)
print(total_deficient_energy)

# Avoid overlapping date indexes
plt.tight_layout()

# Show the plot
plt.show()


new_wind_output=True
if new_wind_output:
    df_wind13_month = df_wind13[df_wind13.index.month == 11]
    df_wind13_month['new wind'] = df_wind13_month['Pdifference'] / 5 + df_wind13_month['NYC_demand']

    # Calculate the area between the curves (excess energy and deficient energy)
    df_wind13_month['area_excess_energy'] = (df_wind13_month['new wind'] - df_wind13_month['NYC_demand']).clip(0) * 0.25
    df_wind13_month['area_deficient_energy'] = (df_wind13_month['NYC_demand'] - df_wind13_month['new wind']).clip(0) * 0.25

    # Plotting the power curves and filled areas
    plt.figure(figsize=(10, 6))
    plt.plot(df_wind13_month.index, df_wind13_month['new wind'], label='Power Output (MW)')
    plt.plot(df_wind13_month.index, df_wind13_month['NYC_demand'], label='NYC Demand (MW)')

    # Filling the areas based on the conditions
    plt.fill_between(df_wind13_month.index, df_wind13_month['new wind'], df_wind13_month['NYC_demand'],
                     where=(df_wind13_month['new wind'] > df_wind13_month['NYC_demand']), interpolate=True, color='green',
                     alpha=0.3)
    plt.fill_between(df_wind13_month.index, df_wind13_month['new wind'], df_wind13_month['NYC_demand'],
                     where=(df_wind13_month['new wind'] < df_wind13_month['NYC_demand']), interpolate=True, color='grey',
                     alpha=0.3)

    # Adding labels and legend with energy values
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')
    plt.title(f'New Power output vs. NYC Demand with eta=5 for November')

    # Calculate and display the total excess and deficient energy for January
    total_excess_energy = (df_wind13_month['area_excess_energy'].sum()) / 1000
    total_deficient_energy = (df_wind13_month['area_deficient_energy'].sum()) / 1000
    plt.legend(
        title=f"Energy (GWh)\nExcess: {total_excess_energy:.2f} GWh\nDeficient: {total_deficient_energy:.2f} GWh")
    print(total_excess_energy)
    print(total_deficient_energy)

    # Avoid overlapping date indexes
    plt.tight_layout()

    # Show the plot
    plt.show()
