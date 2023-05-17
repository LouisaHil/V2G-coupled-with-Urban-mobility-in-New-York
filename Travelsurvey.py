import pandas as pd
import pandas as pd
from datetime import datetime, time, timedelta
# Load the CSV file into a dataframe
#df = pd.read_csv('PLACE_Public.csv')
# Load the CSV file in chunks
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
times = new_df[['SAMPN','VEHNO','DOW','TRP_DEP_HR','TRP_DEP_MIN','TRP_ARR_HR','TRP_ARR_MIN']].copy()

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
# Print the updated DataFrame
print(times)

newfiltered_df=times[['DOW','ID_vehicle','trip_departure_time','trip_arrival_time']].copy()
# sometimes there is duplicates as we filtered out the auto with driver and passenger
newfiltered_df.drop_duplicates(subset=['ID_vehicle', 'trip_departure_time', 'trip_arrival_time'], inplace=True)




# Create a list of 15-minute intervals for five days (Monday to Friday)
time_intervals = pd.date_range('00:00', periods= 24 * 4, freq='15min').time

# Initialize an empty DataFrame to store the counts
counts_df = pd.DataFrame(index=time_intervals, columns=['Count'])

# Iterate over each day of the week
for dow in range(1, 6):  # Assuming DOW values are 1 to 5 for Monday to Friday
    # Filter rows for the current day of the week
    dow_rows = newfiltered_df[newfiltered_df['DOW'] == dow]

    # Iterate over each time interval
    for interval_start in time_intervals:
        interval_end = (datetime.combine(datetime.min, interval_start) + timedelta(minutes=15)).time()

        # Filter rows where no trips overlap with the current interval
        parked_cars = dow_rows[
            ((dow_rows['trip_departure_time'] > interval_end) | (dow_rows['trip_arrival_time'] < interval_start))
        ]

        # Count the number of parked cars
        count = len(parked_cars)

        # Store the count in the counts DataFrame
        counts_df.loc[interval_start, f'Count_DOW_{dow}'] = count

# Print the resulting counts DataFrame
print(counts_df)

# create a new DataFrame with one row for each hour of the day
hours = pd.DataFrame({'hour': range(24)})
merged = pd.merge_asof(times.sort_values('arrival'), hours, left_on='arrival', right_on='hour', direction='backward')
merged = pd.merge_asof(merged.sort_values('departure'), hours, left_on='departure', right_on='hour', direction='forward')

# create a new column indicating whether the person is driving (1) or parked (0) during each hour of the day
merged['driving'] = ((merged['departure'] - merged['arrival']).dt.seconds / 3600) > 0

import matplotlib.pyplot as plt
your_sampn_value=3002177.00000

# select the relevant rows from the merged DataFrame
vehicle_data = merged[merged['SAMPN'] == your_sampn_value]

# plot the driving/parking plot
plt.step(vehicle_data['hour'], vehicle_data['driving'])
plt.xlabel('Hour of the day')
plt.ylabel('Driving (1) or parked (0)')
plt.ylim(-0.1, 1.1)
plt.show()