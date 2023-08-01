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
capacity=9
recompute_dfNY_WIND=True
# Load the CSV file into a Pandas DataFrame
#data = pd.read_csv('Uswinddata2.csv', delimiter=';')
#data=pd.read_csv('USWINDDATA_process_4M.csv',delimiter=';')

df = pd.read_csv(f'USWINDDATA_process_{capacity}M.csv', parse_dates=['Local_time_New_York'], delimiter=';')

df_NY = pd.read_csv('Actual_Load_NY.csv', parse_dates=['RTD End Time Stamp'], delimiter=';')


def preprocess_wind(df):
    df = df.sort_values('Local_time_New_York')
    start_date = df['Local_time_New_York'].min()
    end_date = df['Local_time_New_York'].max()
    freq = '1H'  # 15 minutes
    complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    # Step 3: Find missing datetimes
    missing_indices = []
    expected_datetime = start_date
    for index, row in df.iterrows():
        if row['Local_time_New_York'] != expected_datetime:
            missing_indices.append(index)
            df.at[index, 'Local_time_New_York'] = expected_datetime
        expected_datetime += pd.Timedelta(hours=1)
    df.set_index('Local_time_New_York', inplace=True)
    df = df.drop(['Time_UTC', 'wind_speed'], axis=1)
    df_interpolated = df.resample('15T').interpolate()
    new_index = pd.DatetimeIndex([date.replace(year=2013) for date in df_interpolated.index])
    df_interpolated.index = new_index
    ##### we need to add three rows for the time 23:15, 23:30, 23:45 of the last day of the year
    new_rows = pd.DataFrame([df_interpolated.iloc[-1]] * 3, columns=df_interpolated.columns)
    new_rows.index = pd.date_range(df_interpolated.index[-1], periods=4, freq='15T')[1:]
    df_interpolated = df_interpolated.append(new_rows)
    df_interpolated['Electricity'] = df_interpolated['Electricity'] / 1000
    #df_interpolated['PV'] = df_interpolated['PV'] / 1000
    #df_interpolated = df_interpolated.rename(columns={'Electricity': 'power_output','PV':'PV'})
    df_interpolated = df_interpolated.rename(columns={'Electricity': 'power_output'})
    return df_interpolated


def preprocess_NY(df_NY):
    # Truncate the seconds and convert the column to pandas datetime format
    df_NY['RTD End Time Stamp'] = pd.to_datetime(df_NY['RTD End Time Stamp']).dt.floor('15T')
    # df.drop_duplicates(subset=['RTD End Time Stamp'], keep='first')
    # Set 'RTD End Time Stamp' as the index
    df_NY.set_index('RTD End Time Stamp', inplace=True)
    # Drop duplicates from the index
    df_NY = df_NY[~df_NY.index.duplicated(keep='first')]
    # Generate the expected datetime range with a frequency of 15 minutes
    expected_range = pd.date_range(start=df_NY.index.min(), end=df_NY.index.max(), freq='15T')
    # Reindex with the expected datetime range
    df_NY = df_NY.reindex(expected_range)
    # Interpolate missing values
    df_NY.interpolate(method='pad', inplace=True)
    df_NY = df_NY.drop(df_NY.tail(1).index)
    df_NY = df_NY.drop(['Zone Name', 'Zone PTID'], axis=1)
    new_index = pd.DatetimeIndex([date.replace(year=2013) for date in df_NY.index])
    df_NY.index = new_index
    return df_NY


def dfNY_WIND(df_wind, df_NY, capacity):
    df15 = preprocess_wind(df_wind)
    df_NY = preprocess_NY(df_NY)
    df15['NYC_demand'] = df_NY['RTD Actual Load'].values
    #df15['Pdifference'] = df15['power_output'] +df15['PV'] - df15['NYC_demand']
    df15['Pdifference'] = df15['power_output'] - df15['NYC_demand']
    df15.to_csv(f'Wind_NY_Power{capacity}.csv')


if recompute_dfNY_WIND:
    dfNY_WIND(df, df_NY, capacity)
    df15 = pd.read_csv(f'WindandPV_NY_Power{capacity}.csv', index_col=0, parse_dates=True)
if ~recompute_dfNY_WIND:
    df15 = pd.read_csv(f'Wind_NY_Power{capacity}.csv', index_col=0, parse_dates=True)