import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator

namedata='/Users/louisahillegaart/pythonProject1/Trip_data/trip_data_2.csv'
boolmonthly=True
booldaily=False
#date = '2013-03-01'
#month='2013-03'

def load_extract_save(data,mymonth):
        taxi_df = pd.read_csv(data)
        df_date = taxi_df[taxi_df[' pickup_datetime'].str.contains(mymonth)]
        return taxi_df, df_date
# save the DataFrame to a file
def load_at_date(mydate):
    df_date= pd.read_csv(mydate)
    return df_date

big_df=pd.DataFrame()
for day in range(1, 29):
    date = f'2013-02-{day:02d}'
    month = '2013-02'
    ########################################################################
    # Load the CSV file into a Pandas DataFrame
    taxi_df = pd.read_csv(namedata)
    df_date = taxi_df[taxi_df[' pickup_datetime'].str.contains(date)]

    ########################################################################
    # Load the CSV file into a Pandas DataFrame
     #Convert pickup_datetime and dropoff_datetime to datetime format
    df_date.iloc[:, 5] = pd.to_datetime(df_date.iloc[:, 5])
    df_date.iloc[:, 6] = pd.to_datetime(df_date.iloc[:, 6])
    #print('changedate:',test.tail(20))
    #
    ###this is only to get a rough idea about when the cars are driving the most.
    #if boolmonthly:
       # print(df_date.groupby(df_date[taxi_df.columns[5]].dt.hour)[df_date.columns[7]].sum().sort_values(ascending=False))
    #######################

    ## groupby each medaillon and look at rides within 24 hours :
    unique_ids = df_date['medallion'].unique()
    #print(unique_ids)
    grouped=df_date.groupby('medallion')
    result = {} #create a dictionary where you store your dataframes
    for medallion in unique_ids:
        if medallion in grouped.groups:
            result[medallion] = grouped.get_group(medallion).loc[:, ['medallion', ' pickup_datetime', ' dropoff_datetime']]

    # Create a new DataFrame with all the minutes between the first and last trip
    #print(result)

    # Set the start and end times for the given date
    start_time = pd.to_datetime(date + ' 00:00:00')
    end_time = pd.to_datetime(date + ' 23:59:00')

    # Create a DatetimeIndex with 15-minute frequency
    dt_index = pd.date_range(start=start_time, end=end_time, freq='15min')

    # Convert the DatetimeIndex to a list of tuples of consecutive 15-minute intervals
    time_intervals = [(dt.strftime('%Y-%m-%d %H:%M'), (dt + pd.Timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M'))
                      for dt in dt_index[:-1]]
    # Append the last interval, which ends at the end_time
    time_intervals.append((dt_index[-2].strftime('%Y-%m-%d %H:%M'), end_time.strftime('%Y-%m-%d %H:%M')))
    new_df = pd.DataFrame({'time_interval': time_intervals})

    medallion_num=2013013369

    count=0

    count=count+1
    print(count)
    #print(medallion_num)
    trips=result[medallion_num]
    #print(trips)
    min_time = trips[[' pickup_datetime', ' dropoff_datetime']].values.min()
    max_time = trips[[' pickup_datetime', ' dropoff_datetime']].values.max()
    #all_minutes = pd.date_range(pd.Timestamp(min_time).floor('min'), pd.Timestamp(max_time).ceil('min'), freq='min')
    #timeline = pd.DataFrame({'datetime': all_minutes})
    trips['duration'] = (trips[' dropoff_datetime'] - trips[' pickup_datetime']).dt.total_seconds() / 60.0


    #### finding out the duration in between the rides that we are considering as downtimes
    downtime_start=trips[' dropoff_datetime']
    downtime_end=trips[' pickup_datetime']
    ## the start 00:00 end the end time 23:59:00 need to be appended
    new_row = pd.Series([date + ' 00:00:00'], name='date')
    new_row2 = pd.Series([date + ' 23:59:00'], name='date')
    # append the new row to the Series at the first position using iloc
    downtime_start = new_row.append(downtime_start.iloc[:]).reset_index(drop=True)
    downtime_end= downtime_end.append(new_row2).reset_index(drop=True)
    ## calculate the duration
    duration_offtime=(downtime_end - downtime_start).dt.total_seconds() / 60.0

    #### filter out the time intervals for which the duration is bigger than 1 hour.
    mask = duration_offtime > 60
    # filter the time differences using the mask
    filtered_diff = duration_offtime[mask]
    inter_downtime_start = downtime_start[mask]
    inter_downtime_end = downtime_end[mask]
    new_interval = pd.Series([(start, end) for start, end in zip(inter_downtime_start, inter_downtime_end)])

    mergedlist=[]
    # Iterate over each interval in new_interval and check for overlap with time_intervals
    if len(new_interval)>0:      ## some car id are never free for 1 hour over a day
        for new_int in new_interval:
           overlap_list = []
           for time_int in time_intervals:
               start_time_int = pd.to_datetime(time_int[0])
               end_time_int = pd.to_datetime(time_int[1])
               tryout1=new_int[0]
               tryout2 = new_int[1]
               if (new_int[0] >= start_time_int and new_int[0] < end_time_int) or \
                  (new_int[1] > start_time_int and new_int[1] <= end_time_int) or (new_int[0] <= start_time_int and new_int[1] >= end_time_int):
                   overlap = 1
               else:
                   overlap=0
               overlap_list.append(overlap)
           mergedlist.append(overlap_list)
        result2 = [sum(x) for x in zip(*mergedlist)]
        result2 = pd.Series(result2, name='medaillon')
    else:
        result2=[0 for i in range(len(time_intervals))]
        result2 = pd.Series(result2, name='medaillon')
    # Create a DataFrame with the time_intervals and overlap_list
    df = pd.DataFrame({'time_interval': time_intervals, 'medaillon_num': result2})
    new_df = pd.concat([new_df, result2], axis=1)

    row_sums = new_df.iloc[:, 1:].sum(axis=1)
    #new_df2 = pd.DataFrame({'time_interval': time_intervals})
    #row_sums.to_csv('tryout1', index=False)
    #new_df2=pd.concat([new_df2, row_sums], axis=1)
    big_df=pd.concat([big_df, row_sums], axis=1)  
    print(big_df)
    #new_df2.to_csv('NbofEVS',index=False)
    #print('number of cars available',new_df2)
time_df=pd.DataFrame({'time_interval': time_intervals})
big_df=pd.concat([time_df, big_df], axis=1)
#big_df.to_csv('NbofEvs_march', index=False)

###### vectorized option ################










plotting=True
if plotting:
    ################# plot a timeline
    # create a figure and axis
    fig, ax = plt.subplots()

    # set the title
    ax.set_title(f'Taxi Drives: Medaillon {medallion_num}')

    # set the y-axis label
    ax.set_ylabel('Taxi')

    # set the x-axis label
    ax.set_xlabel('Time')

    # set the y-axis limits
    ax.set_ylim(0, 2)
    ax.set_yticks(range(0, 3, 1))

    # plot a horizontal line for each taxi trip
    for i, row in trips.iterrows():
        y = i
        xstart = row[' pickup_datetime']
        xend = row[' dropoff_datetime']
        duration = row['duration']
        if duration > 0:
            ax.hlines(y=1, xmin=xstart, xmax=xend, linewidth=100)
        else:
            ax.plot(xstart, y, 'o', markersize=12)

    # format the x-axis ticks and labels
    #ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    #ax.tick_params(axis='x', rotation=90, labelsize=6)
    # format the x-axis ticks and labels
    # set the x-axis major tick locator to an HourLocator
    ax.xaxis.set_major_locator(HourLocator())

    # set the x-axis minor tick locator to a MinuteLocator
    ax.xaxis.set_minor_locator(MinuteLocator(interval=15))

    # set the tick positions and labels on the x-axis
    hours = pd.date_range(start=trips[' pickup_datetime'].min().floor('H'), end=trips[' dropoff_datetime'].max().ceil('H'), freq='H')
    print(hours)
    ax.set_xticks(hours)
    ax.set_xticklabels(hours.strftime('%H:%M'))
    ax.tick_params(axis='x', rotation=90, labelsize=8)  # adjust label size here

    # display the plot
    plt.show()