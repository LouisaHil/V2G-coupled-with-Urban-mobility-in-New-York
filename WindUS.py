import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.dates as mdates
import numpy as np
from scipy.integrate import simps, quad

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('Uswinddata2.csv', delimiter=';')

# Replace thousands and decimal separators in the Electricity column
data['Electricity'] = data['Electricity'].str.replace('.', '').str.replace(',', '.')

# Convert the Local_time_New_York column to datetime format
data['Local_time_New_York'] = pd.to_datetime(data['Local_time_New_York'])

# Remove duplicate values from the Local_time_New_York column
data = data.loc[~data['Local_time_New_York'].duplicated()]

# Create a new DataFrame with only the Electricity and Local_time_New_York columns
data_new = data[['Electricity', 'Local_time_New_York']]

# Convert the Electricity column to a NumPy array
electricity_arr = data_new['Electricity'].to_numpy()

# Create an interpolation function for the Electricity data
interp_func = interp1d(data_new['Local_time_New_York'].astype(np.int64) // 10 ** 9, electricity_arr, kind='cubic')

# Create a new time array with finer resolution for plotting
new_times = pd.date_range(data_new['Local_time_New_York'].min(), data_new['Local_time_New_York'].max(), freq='15T')

# Evaluate the interpolation function at the new times
interp_vals = interp_func(new_times.astype(np.int64) // 10 ** 9)

# Plot the interpolated data
plt.plot(new_times, interp_vals/1000000)
plt.xlabel('Local Time (New York)')
plt.ylabel('Electricity Mw')
plt.title('Electricity consumption ')
plt.show()