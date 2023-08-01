import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta


Trip_recompute_private_mobility=True
expanding_to_month=False
expanding_to_year=False


if Trip_recompute_private_mobility:
    # #Load the CSV file in chunks
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

    newfiltered_df=times[['DOW','ID_vehicle','OTPURP_AGG','DTPURP_AGG','trip_departure_time','trip_arrival_time','TRPDUR','TRIPDIST']].copy()
    # sometimes there is duplicates as we filtered out the auto with driver and passenger
    newfiltered_df.drop_duplicates(subset=['ID_vehicle', 'trip_departure_time', 'trip_arrival_time'], inplace=True)


    ### calculation the amount of cars parked at a given period

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

    print(counts_df)
    unique_medallions = newfiltered_df['ID_vehicle'].unique()
    print(len(unique_medallions))
    # Start date
    date = '2013-01-01'
    # Function to add date to time
    def add_date_to_time(row, date, column_name):
        time_str = row[column_name].strftime('%H:%M:%S') # Convert time to string
        return datetime.strptime(date + ' ' + time_str, '%Y-%m-%d %H:%M:%S')

    # Add date to 'trip_departure_time' and 'trip_arrival_time'
    newfiltered_df['trip_departure_time'] = newfiltered_df.apply(add_date_to_time, args=(date, 'trip_departure_time',), axis=1)
    newfiltered_df['trip_arrival_time'] = newfiltered_df.apply(add_date_to_time, args=(date, 'trip_arrival_time',), axis=1)

    # Define function to duplicate entries for each day of the month
    def duplicate_entries_for_each_day(df):
        all_days = pd.date_range(date, periods=30, freq='D')
        df_duplicates = pd.concat([df.assign(trip_departure_time=df['trip_departure_time'] + timedelta(days=i),
                                             trip_arrival_time=df['trip_arrival_time'] + timedelta(days=i))
                                   for i in range(1, len(all_days))], ignore_index=True)
        return df_duplicates

    # Group by vehicle ID and duplicate entries for each vehicle
    newfiltered_df_duplicates = pd.concat([duplicate_entries_for_each_day(group) for _, group in newfiltered_df.groupby('ID_vehicle')], ignore_index=True)

    # Combine the initial dataframe and duplicates
    newfiltered_df = pd.concat([newfiltered_df, newfiltered_df_duplicates], ignore_index=True)

    # Drop 'DOW' column
    newfiltered_df = newfiltered_df.drop('DOW', axis=1)
    newfiltered_df = newfiltered_df.sort_values(by=['ID_vehicle', 'trip_departure_time'])
    # Rename columns
    newfiltered_df = newfiltered_df.rename(columns={
        'ID_vehicle': 'medallion',
        'trip_departure_time': 'pickup_datetime',
        'trip_arrival_time': 'dropoff_datetime',
        'TRPDUR': 'duration',
    })
    newfiltered_df.to_csv('Private_trip_1.csv', index=False)
##### expand the dataset to the whole month
if expanding_to_month:
    ### change path to rerun code
    #Read the CSV file
    namedata='/Users/louisahillegaart/pythonProject1/Private_trip_1.csv'
    def load_and_preprocess_data(namedata, month):
        taxi_df = pd.read_csv(namedata, parse_dates=['pickup_datetime', 'dropoff_datetime'])
        #taxi_df = pd.read_csv(namedata, parse_dates=[5, 6])
        taxi_df.columns = taxi_df.columns.str.strip()
        taxi_df = taxi_df[(taxi_df['pickup_datetime'].dt.strftime('%Y-%m') == month) & (
                    taxi_df['dropoff_datetime'].dt.strftime('%Y-%m') == month)]
        taxi_df['duration'] = (taxi_df['dropoff_datetime'] - taxi_df['pickup_datetime']).dt.total_seconds() / 60
        taxi_df = taxi_df[['medallion', 'pickup_datetime', 'dropoff_datetime', 'duration']]
        duration_df = taxi_df[['medallion', 'duration']]
        return taxi_df, duration_df


    def calculate_availability(df, month):
        unique_medallions = df['medallion'].unique()
        time_intervals = pd.date_range(f"{month}-01 00:00:00", f"{'2013-02'}-01 00:00:00", freq='15min')
        availability_matrix = np.ones((len(time_intervals) - 1, len(unique_medallions)), dtype=int)

        for idx, medallion in enumerate(unique_medallions):
            medallion_df = df[df['medallion'] == medallion].copy()

            # Add the initial off-period
            start_row = pd.Series({
                'pickup_datetime': pd.Timestamp(f"{month}-01 00:00:00"),
                'dropoff_datetime': medallion_df.iloc[0]['pickup_datetime']
            })
            medallion_df = pd.concat([start_row, medallion_df], ignore_index=True)

            # Create the off_intervals DataFrame
            off_intervals = medallion_df[['dropoff_datetime']].rename(columns={'dropoff_datetime': 'start_offtime'})
            off_intervals['end_offtime'] = medallion_df['pickup_datetime'].shift(-1)
            off_intervals['duration'] = (off_intervals['end_offtime'] - off_intervals[
                'start_offtime']).dt.total_seconds() / 60
            off_intervals = off_intervals[off_intervals['duration'] > 60]

            for _, row in medallion_df.iterrows():
                overlap = (time_intervals[:-1] < row['dropoff_datetime']) & (time_intervals[1:] > row['pickup_datetime'])
                availability_matrix[overlap, idx] = 0

            availability_df = pd.DataFrame(availability_matrix, columns=unique_medallions, index=time_intervals[:-1])
            availability_df['total'] = availability_df.sum(axis=1)
            availability_df.reset_index(inplace=True)
            availability_df.rename(columns={'index': 'time_interval'}, inplace=True)
        return availability_df


    month = '2013-01'  # you need to change the last date in line 16 of this code depending on the month
    taxi_df, duration_df = load_and_preprocess_data(namedata, month)
    availability_df = calculate_availability(taxi_df, month)
    availability_df.to_csv('Private_NbofEvs_1.csv', index=False)
    availability_df[['time_interval', 'total']].to_csv('PrivateTotalNbofEvs_1.csv', index=False)
##### expand the dataset to the whole year
if expanding_to_year:
    ################### This part is used for expanding the dataset to one year.
    # Read the CSV file
    df = pd.read_csv('Private_NbofEvs_1.csv')
    # Convert the datetime column to datetime format
    df['time_interval'] = pd.to_datetime(df['time_interval'])

    # Define the start and end dates
    start_date = pd.Timestamp('2013-01-01')
    end_date = pd.Timestamp('2013-12-31 23:45:00')

    # Create an empty DataFrame to store the expanded dataset
    expanded_df = pd.DataFrame()

    # Loop through each month
    current_date = start_date
    while current_date <= end_date:
        # Calculate the last day of the current month
        last_day = pd.Timestamp(current_date.year, current_date.month, pd.DatetimeIndex([current_date]).days_in_month[0],
                                23, 45)

        # Filter the original dataframe for the current month
        current_month_df = df[(df['time_interval'].dt.month == current_date.month)]

        # Filter the current month data for the current month's number of days
        current_month_days = last_day.day

        # Repeat the current month's data for the number of days
        repeated_df = pd.concat([current_month_df] * current_month_days, ignore_index=True)

        # Update the datetime values based on the current month
        repeated_df['time_interval'] = pd.date_range(start=current_date, periods=len(repeated_df), freq='15min')

        # Append the current month data to the expanded dataframe
        expanded_df = expanded_df.append(repeated_df)

        # Move to the next month
        current_date = last_day + pd.offsets.Day(1)
        # Stop if the current_date exceeds the end_date
        if current_date > end_date:
            break

    # Reset the index and rename the column
    expanded_df.reset_index(drop=True, inplace=True)
    expanded_df.rename(columns={'time_interval': 'time_interval'}, inplace=True)
    # Truncate the expanded dataset to the desired length
    expanded_df = expanded_df.iloc[:35040]
    # Save the expanded dataframe to a new CSV file
    expanded_df.to_csv('expanded_dataset.csv', index=False)
