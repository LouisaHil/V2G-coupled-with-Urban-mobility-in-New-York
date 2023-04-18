
import pandas as pd


namedata='/Users/louisahillegaart/pythonProject1/Trip_data/trip_data_1.csv'


def load_and_preprocess_data(namedata, month):
     taxi_df = pd.read_csv(namedata, parse_dates=[5, 6])
     taxi_df.columns = taxi_df.columns.str.strip()
     taxi_df = taxi_df[(taxi_df['pickup_datetime'].dt.strftime('%Y-%m') == month) & (taxi_df['dropoff_datetime'].dt.strftime('%Y-%m') == month)]
     taxi_df['duration'] = (taxi_df['dropoff_datetime'] - taxi_df['pickup_datetime']).dt.total_seconds() / 60
     taxi_df = taxi_df[['medallion', 'pickup_datetime', 'dropoff_datetime', 'duration']]
     duration_df=taxi_df[['medallion', 'duration']]
     return taxi_df, duration_df


df = pd.read_csv('new_NbofEvs_1.csv') ## depends on the month chosen
# Convert the timestamp column to pandas datetime format
df['time_interval'] = pd.to_datetime(df['time_interval'])
see=df['2013013426']
start_date = '2013-01-01'
end_date = '2013-01-07'
medaillon_number=2013000001
mask = (df['time_interval'] >= start_date) & (df['time_interval'] <= end_date)
df_filtered = df[mask]

# get the duration of the trips for each medaillon

duration_df=load_and_preprocess_data(namedata,'2013-01')[1]

grouped_duration= duration_df[duration_df['medallion'] == medaillon_number]


print(grouped_duration)
