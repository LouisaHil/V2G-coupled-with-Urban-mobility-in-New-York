import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import numpy as np
import scipy
from datetime import datetime, time, timedelta
from sklearn.mixture import GaussianMixture

 #Load the CSV file in chunks
df_chunks = pd.read_csv('UNLINKED_Public.csv',delimiter=';', chunksize=1000,error_bad_lines=False)

# Initialize an empty list to store the filtered dataframes
filtered_dfs = []

# Loop through each chunk of data
for df in df_chunks:
    # Filter out rows with "New York" in the "CITY" column
    filtered_df = df[df['ONYC'] == 1]
    filtered_df = filtered_df[filtered_df['DNYC'] == 1]

    # Add the filtered dataframe to the list
    filtered_dfs.append(filtered_df)

# Concatenate the list of dataframes into a single dataframe
df = pd.concat(filtered_dfs, ignore_index=True)


# Extract specific columns into a new dataframe
columns_to_extract = ['SAMPN','VEHNO', 'GTYPE', 'TRIPNO','LTRIPNO','DOW','ULTMODE','LTMODE_AGG','MODE_SAMP','PRKTY','TRP_DEP_HR','TRP_DEP_MIN','TRP_ARR_HR','TRP_ARR_MIN','TRPDUR','TRIPDIST']
new_df = df[columns_to_extract]
# Filter rows where 'ULTMODE' is 5 or 6
new_df = df[(df['ULTMODE'] == 5) | (df['ULTMODE'] == 6)]
# create a new DataFrame with start and end times
times = new_df[['SAMPN','VEHNO','DOW','OTPURP_AGG','DTPURP_AGG','TRP_DEP_HR','TRP_DEP_MIN','TRP_ARR_HR','TRP_ARR_MIN','TRPDUR','TRIPDIST']].copy()

# Convert 'trip_arrival_hour' and 'trip_arrival_min' columns to integers
times['TRP_ARR_HR'] = times['TRP_ARR_HR'].astype(int)
times['TRP_ARR_MIN'] = times['TRP_ARR_MIN'].astype(int)

# Convert 'trip_arrival_hour' and 'trip_arrival_min' columns to integers
times['TRP_DEP_HR'] = times['TRP_DEP_HR'].astype(int)
times['TRP_DEP_MIN'] = times['TRP_DEP_MIN'].astype(int)

# Adjust hour values greater than or equal to 24
times.loc[df['TRP_ARR_HR'] >= 24, 'TRP_ARR_HR'] = 23
times.loc[df['TRP_ARR_HR'] >= 24, 'TRP_ARR_MIN'] = 59

# Adjust hour values greater than or equal to 24
times.loc[df['TRP_DEP_HR'] >= 24, 'TRP_DEP_HR'] = 23
times.loc[df['TRP_DEP_HR'] >= 24, 'TRP_DEP_MIN'] = 59

# Create a new column 'trip_arrival_time' by combining 'trip_arrival_hour' and 'trip_arrival_min'
times['trip_arrival_time'] = pd.to_datetime(times['TRP_ARR_HR'].astype(str).str.zfill(2) + ':' + times['TRP_ARR_MIN'].astype(str).str.zfill(2), format='%H:%M').dt.time
times['trip_departure_time'] = pd.to_datetime(times['TRP_DEP_HR'].astype(str).str.zfill(2) + ':' + times['TRP_DEP_MIN'].astype(str).str.zfill(2), format='%H:%M').dt.time

# Convert 'SAMPN' and 'VEHNO' columns to strings
times['SAMPN'] = times['SAMPN'].astype(str)
times['VEHNO'] = times['VEHNO'].astype(str)

# Concatenate 'SAMPN' and 'VEHNO' columns into a new column 'combined_number'
times['ID_vehicle'] = df['SAMPN'] + df['VEHNO']

newfiltered_df=times[['DOW','ID_vehicle','trip_departure_time','trip_arrival_time','TRPDUR','TRIPDIST']].copy()
# sometimes there is duplicates as we filtered out the auto with driver and passenger
newfiltered_df.drop_duplicates(subset=['ID_vehicle', 'trip_departure_time', 'trip_arrival_time'], inplace=True)

# Assuming your dataframe is 'times'
times['trip_arrival_time'] = pd.to_datetime(times['TRP_ARR_HR'].astype(str).str.zfill(2) + ':' + times['TRP_ARR_MIN'].astype(str).str.zfill(2), format='%H:%M').dt.time
times['trip_departure_time'] = pd.to_datetime(times['TRP_DEP_HR'].astype(str).str.zfill(2) + ':' + times['TRP_DEP_MIN'].astype(str).str.zfill(2), format='%H:%M').dt.time

newfiltered_df = times[['DOW','ID_vehicle','trip_departure_time','trip_arrival_time','TRPDUR','TRIPDIST']].copy()

newfiltered_df['departure_week_time'] = (newfiltered_df['DOW']-1) * 24 + newfiltered_df['trip_departure_time'].apply(lambda x: x.hour + x.minute / 60)
newfiltered_df['arrival_week_time'] = (newfiltered_df['DOW']-1) * 24 + newfiltered_df['trip_arrival_time'].apply(lambda x: x.hour + x.minute / 60)

# Filter out for a single day (DOW = 1)
filtered_data = newfiltered_df[newfiltered_df['DOW'] == 1]

# Binning the 'departure_week_time' into 15-minute intervals
# Divide the 'departure_week_time' by 4 to convert 15-minute intervals into hours
filtered_data['Binned Departure Time'] = (filtered_data['departure_week_time'] // (15/60)) / 4

def kerneldensity(filtered_data):
    # Fit the kernel density model
    kde = KernelDensity(kernel='gaussian', bandwidth=0.6)
    kde.fit(filtered_data[['Binned Departure Time']])

    # Generate new samples
    num_samples = 490000  # Number of cars in the larger fleet
    new_samples = kde.sample(num_samples)

    # Convert samples to a DataFrame
    new_samples_df = pd.DataFrame(new_samples, columns=['Binned Departure Time'])

    # Remove negative values
    new_samples_df = new_samples_df[new_samples_df['Binned Departure Time'] >= 0]

    # Bin the generated samples into 15-minute intervals
    new_samples_df['Binned Departure Time'] = (new_samples_df['Binned Departure Time']*4).astype(int) // 1 / 4

    # Count the number of samples in each bin
    departure_counts = new_samples_df.groupby('Binned Departure Time').size()
    return departure_counts

def betadensity(filtered_data):
    # Scale departure times to the range [0, 1]
    scaled_departure_times = filtered_data['Binned Departure Time'] / 24

    # Fit Beta distribution to data
    alpha, beta, _, _ = scipy.stats.beta.fit(scaled_departure_times)

    # Generate new samples
    num_samples = 490000  # Number of cars in the larger fleet
    new_samples = scipy.stats.beta.rvs(alpha, beta, size=num_samples)

    # Scale samples back to the range [0, 23.75]
    new_samples *= 23.75

    # Round to nearest 15-minute interval
    new_samples = np.round(new_samples * 4) / 4

    # Count the number of samples in each bin
    departure_counts = np.histogram(new_samples, bins=np.arange(0, 24, 1 / 4))
    return departure_counts

result=betadensity(filtered_data)
print(result)





## Group by departure_week_time and arrival_week_time and count the number of vehicles
#### for thedistribution i only need to look at one day :
#departure_counts = newfiltered_df[newfiltered_df['DOW'] == 1].groupby(newfiltered_df['departure_week_time'] // (15/60)).count()
#arrival_counts = newfiltered_df[newfiltered_df['DOW'] == 1].groupby(newfiltered_df['arrival_week_time'] // (15/60)).count()
##departure_counts = newfiltered_df[newfiltered_df['DOW'] == 1].groupby((newfiltered_df['departure_week_time'] // (15/60)).astype(int))
#
#
## Reshape the data to fit the model
#departure_counts_reshaped = departure_counts['ID_vehicle'].values.reshape(-1, 1)
#
# Create a Gaussian Mixture Model with 2 components (for bimodality)
#gmm = GaussianMixture(n_components=2)

# Fit the model to the data
#gmm.fit(departure_counts_reshaped)

# Generate new samples (scale up the data)
#scaled_departure_counts = gmm.sample(700000)[0]




# Create a figure and axes
fig, ax = plt.subplots(figsize=[15,10])

# Line plot for departures
ax.plot(departure_counts.index * (15/60), departure_counts['ID_vehicle'], label='Departures')

# Line plot for arrivals
ax.plot(arrival_counts.index * (15/60), arrival_counts['ID_vehicle'], label='Arrivals')

# Label and show the plot
ax.set_xlabel('Time of Week (Hours)')
ax.set_ylabel('Number of Trips')
ax.set_xticks(range(0, 24, 24))
ax.set_xticklabels(['Monday'])
ax.legend(loc='upper left')
ax.set_title('Number of Departures and Arrivals Over Time')
plt.grid(True)

plt.show()
# Plot the histogram of the original data
plt.hist(departure_counts['ID_vehicle'], bins=50, alpha=0.5, label='Original')

# Plot the histogram of the new data
#plt.hist(scaled_departure_counts, bins=50, alpha=0.5, label='Scaled')

# Add title and labels
plt.title("Comparison of Original and Scaled Departure Counts")
plt.xlabel("Departure Count")
plt.ylabel("Frequency")
plt.legend(loc='upper right')

# Show the plot

plt.show()




