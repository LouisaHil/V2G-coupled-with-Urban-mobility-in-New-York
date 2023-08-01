import numpy as np
import pandas as pd
scaling=10
##df=pd.read_csv('Private_NbofEvs_1.csv', index_col=0)
df= pd.read_csv('expanded_dataset.csv', index_col='time_interval')
#df=pd.read_csv('new_NbofEvs_1.csv', index_col=0)
# Assume df is your DataFrame, with datetime index and car_id as columns
# Calculate the probability of being parked at each 15-minute interval
#df_soc_min.to_csv('2_scaleSOCmin_data.csv')
#df1= pd.read_csv('2SyntheticData.csv', index_col='time_interval')
#df_cars = pd.read_csv('2_scaleSOCmin_data.csv', index_col='time_interval')
def scaledata(df):
    df = df.iloc[:, :-1]

    prob_parked = df.mean(axis=1)

    np.random.seed(0)
    synthetic_data = []
     # scale for each row independently
    for prob in prob_parked:
        synthetic_interval_data = np.random.choice([0, 1], size=df.shape[1] * scaling, p=[1 - prob, prob])
        synthetic_data.append(synthetic_interval_data)

    synthetic_data = np.array(synthetic_data)

    np.random.seed(0)
    columns = np.random.randint(1, 1000, size=df.shape[1] * scaling)
    index = df.index

    df_cars = pd.DataFrame(synthetic_data, index=index, columns=columns)

    return df_cars


recomputescaled=True
if recomputescaled==True :
    df_cars=scaledata(df)
    df_cars.to_csv(f'{scaling}SyntheticData.csv')
    print('done creating dataset')

######### SOC MIN #############

#df_cars = pd.read_csv('5SyntheticData.csv', index_col='time_interval')
#df_cars.index = pd.to_datetime(df_cars.index)
#df_cars = pd.read_csv('SyntheticJanuaryData.csv', parse_dates=['time_interval'], index_col='time_interval', date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
#prob_parked = df_cars.mean(axis=1)
#df = df.iloc[:, :-1]
#small_prob_parked = df.mean(axis=1)


# Parameters
battery_capacity = 70  # kWh
charging_rate = 50  # Assumed constant charging rate in kW -> kWh/15min = 50/4
initial_soc_min = 0  # initial SOCmin at 2013-01-31 23:45
energy_loss_driving = 0.75   # This will depend on your actual data


def SOC_min_vect(column):
    series_reversed = column.iloc[::-1]
    groups = (series_reversed != series_reversed.shift()).cumsum()
    soc_min = (series_reversed == 0).groupby(groups).cumsum()
    soc_min = soc_min * energy_loss_driving / battery_capacity
    soc_min = soc_min[::-1]
    return soc_min


# Apply the function to each column of the DataFrame
SOC_min_recompute=False
if SOC_min_recompute==True:
    df_soc_min = df_cars.apply(SOC_min_vect)
    df_soc_min.to_csv(f'{scaling}_scaleSOCmin_data.csv')
    print(df_soc_min)

