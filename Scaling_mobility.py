import numpy as np
import pandas as pd
scaling=20
df=pd.read_csv('Private_NbofEvs_1.csv', index_col=0)
#df=pd.read_csv('new_NbofEvs_1.csv', index_col=0)
# Assume df is your DataFrame, with datetime index and car_id as columns
# Calculate the probability of being parked at each 15-minute interval


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


recomputescaled=False
if recomputescaled==True :
    df_cars=scaledata(df)
    df_cars.to_csv('20SyntheticJanuaryData.csv')

######### SOC MIN #############
df_cars = pd.read_csv('20SyntheticJanuaryData.csv', index_col='time_interval')
df_cars.index = pd.to_datetime(df_cars.index)
#df_cars = pd.read_csv('SyntheticJanuaryData.csv', parse_dates=['time_interval'], index_col='time_interval', date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
prob_parked = df_cars.mean(axis=1)
df = df.iloc[:, :-1]
small_prob_parked = df.mean(axis=1)


print(df_cars.index.min())
print(df_cars.index.max())
# Parameters
battery_capacity = 70  # kWh
charging_rate = 50  # Assumed constant charging rate in kW -> kWh/15min = 50/4
initial_soc_min = 0  # initial SOCmin at 2013-01-31 23:45
energy_loss_per_15min_while_driving = 0.75  # This will depend on your actual data


def SOC_min_vect(column):
    series_reversed = column.iloc[::-1]
    groups = (series_reversed != series_reversed.shift()).cumsum()
    soc_min = (series_reversed == 0).groupby(groups).cumsum()
    soc_min = soc_min * energy_loss_per_15min_while_driving / battery_capacity
    soc_min = soc_min[::-1]
    return soc_min


# Apply the function to each column of the DataFrame
SOC_min_recompute=True
if SOC_min_recompute==True:
    df_soc_min = df.apply(SOC_min_vect)
    df_soc_min.to_csv('Feb_SOCmin_data.csv')
    print(df_soc_min)


df = df.iloc[:, :-1]
prob_parked = df.mean(axis=1)
size=1300000
# Set the desired number of parked/driving cars for each time instance
desired_count = prob_parked * size  # Multiply by the total number of data points in df_cars


# Set the initial dataset size and accuracy threshold
initial_dataset_size = 2694  # Adjust as needed
accuracy_threshold = 0.95  # Adjust as needed

# Initialize the smallest dataset and scaling factor
smallest_dataset = df_cars.sample(n=initial_dataset_size, replace=True)  # Randomly select initial dataset
smallest_scaling_factor = (desired_count / smallest_dataset.sum(axis=1)).max()
# Calculate the initial accuracy
accuracy = 1.0 - abs(
    (desired_count.sum() - smallest_scaling_factor * smallest_dataset.sum().sum()) / desired_count.sum())

# Iteratively increase the dataset size while maximizing the accuracy
while accuracy < accuracy_threshold:
    # Incrementally increase the dataset size
    smallest_dataset = smallest_dataset.append(df_cars.sample(n=initial_dataset_size, replace=True))

    # Recalculate the scaling factor and accuracy
    smallest_scaling_factor = desired_count.sum() / smallest_dataset.sum().sum()
    accuracy = 1.0 - abs(
        (desired_count.sum() - smallest_scaling_factor * smallest_dataset.sum().sum()) / desired_count.sum())

# Print the final dataset size and accuracy
final_dataset_size = len(smallest_dataset)
print(f"Final Dataset Size: {final_dataset_size}")
print(f"Accuracy: {accuracy}")