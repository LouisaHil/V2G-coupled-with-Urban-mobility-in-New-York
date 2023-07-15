import pandas as pd
import numpy as np
import random
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


month=1
df_cars = pd.read_csv('expanded_dataset.csv', index_col='time_interval')
df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')

def load_and_extract_month(df_cars,month):
    df_cars.index = pd.to_datetime(df_cars.index)
    df_cars = df_cars.sort_values(by='time_interval')
    df_cars = df_cars[df_cars.index.month == month]
    df_cars = df_cars.drop('total', axis=1)
    return df_cars


df_cars_month =load_and_extract_month(df_cars ,month)
#df_15 =load_and_extract_month(df15 ,month)
#df_SOC_min_month = load_and_extract_month(df_SOC_min, month)
# Get the first index from df_cars_month
first_index = df_cars_month.index[0]

# Initialize randomly the SOC for one row
df_SOC = pd.DataFrame(
    np.random.choice(np.linspace(0.1, 1, 10), size=(1, df_cars_month.shape[1])),
    index=[first_index],
    columns=df_cars_month.columns
)


def get_SOC_list_t(df_SOC, relevant_columns_indices, t, scaling_factor):
    # Get the relevant columns from df_SOC
    df_relevant = df_SOC.loc[t, relevant_columns_indices]

    # Get the SOC values you want to replicate
    SOC_values_to_replicate = df_relevant.values

    # Get the indices to replicate
    indices_to_replicate = df_relevant.index

    # Repeat the values scaling_factor times
    replicated_SOC_values = np.repeat(SOC_values_to_replicate, scaling_factor)

    # Repeat the indices scaling_factor times
    replicated_indices = np.repeat(indices_to_replicate, scaling_factor)

    # Return a Series with replicated SOC values and replicated indices
    return pd.Series(replicated_SOC_values, index=replicated_indices)

def get_replicate_df(df_cars_month, relevant_column_indices, scaling_factor):
    # Initialize an empty DataFrame to store the replicated columns
    replicated_dfmonth = pd.DataFrame()
    # Initialize an empty dictionary to store the mapping
    column_mapping = {}
    # Iterate over the relevant column indices
    for col_name in relevant_column_indices:
        # Get the column values for the current column name
        col_values = df_cars_month[col_name]

        # Create new column names for the replicated columns
        new_col_names = [f"{col_name}{i+1}" for i in range(scaling_factor)]
        # Store the column mapping in the dictionary
        column_mapping[col_name] = new_col_names
        # Create new columns with the replicated values and updated column names
        for new_col_name in new_col_names:
            replicated_dfmonth[new_col_name] = col_values

    # Set the index of the replicated DataFrame to match the original df_cars_month index
    replicated_dfmonth.index = df_cars_month.index

    return replicated_dfmonth, column_mapping


def critical_SOC_power_calculator(t,parked_prev, critical, rate, battery_capacity, df_SOC_,df_Power):
    df_SOC_.loc[t, critical & parked_prev] = df_SOC_.loc[t-pd.DateOffset(minutes=15), critical & parked_prev] + (rate*delta_t)/battery_capacity
    #df_Power.loc[t, critical & parked_prev] = -(rate * scaling_factor)/1000
    #df_SOC.loc[t, critical & parked_prev] = df_SOC.shift().loc[t, critical & parked_prev]
    #df_Power.loc[t, critical & parked_prev] = 0
    # df_CarCounts.loc[t, 'Critical_noscale'] = critical.sum()
# Initialize an empty dictionary
def charge(df_SOC_,rate,scaling_factor,delta_t,battery_capacity,df_Power,df_CarCounts):
    SOC_sorted = df_SOC_.loc[t - pd.DateOffset(minutes=15), condition2 & ~critical & parked_prev].sort_values(
        ascending=True)
    new_Power = pd.Series((-rate * scaling_factor) / 1000, index=SOC_sorted.index)
    cumulative_Power = new_Power.cumsum()
    # Determine the number of cars for which we need to calculate the Power
    Nb = cumulative_Power[-cumulative_Power < abs(Pdifference_left)].count()
    df_CarCounts.loc[t, 'Charging_noscale'] = Nb
    new_row[new_Power.index[:Nb]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[:Nb]] + (
                rate * delta_t) / battery_capacity
    relevant_columns = new_Power.index[:Nb]
    #df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]
    ##update soc and power for all remaining cars
    new_row[new_Power.index[Nb:]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    #df_Power.loc[t, new_Power.index[Nb:]] = 0
    return relevant_columns



def discharge(df_SOC_,rate,scaling_factor,delta_t,battery_capacity,df_Power,df_CarCounts):
    # available_discharge = (df_SOC.shift().loc[t,parked_prev] != 0 & ~critical)
    SOC_sorted = df_SOC_.loc[t - pd.DateOffset(minutes=15), condition & ~critical & parked_prev].sort_values(ascending=False)
    # Calculate new Power for all parked, not critical cars
    # power given in KW we want it in MW need to dived by 1000 to get MW
    new_Power = pd.Series((rate * scaling_factor) / 1000, index=SOC_sorted.index)
    cumulative_Power = new_Power.cumsum()

    # Determine the number of cars for which we need to calculate the Power
    Nb = cumulative_Power[cumulative_Power <= abs(Pdifference_left)].count() + 1
    df_CarCounts.loc[t, 'Discharging_noscale'] = Nb
    new_row[new_Power.index[:Nb]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[:Nb]] - (
            rate * delta_t) / battery_capacity
    relevant_columns = new_Power.index[:Nb]
    #df_Power.loc[t, new_Power.index[:Nb]] = new_Power.loc[new_Power.index[:Nb]]

    new_row[new_Power.index[Nb:]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    #df_Power.loc[t, new_Power.index[Nb:]] = 0
    return relevant_columns

SOC_dict = {}
for t in df_cars_month.index[1:]:
    # If SOC_dict is not empty and t-1 exists in SOC_dict, convert the last entry to a dataframe
    if SOC_dict and (t - pd.DateOffset(minutes=15)) in SOC_dict:
        # Convert your datetime t-1 to a dataframe
        df_SOC_ = pd.DataFrame(SOC_dict[t - pd.DateOffset(minutes=15)], index=[t - pd.DateOffset(minutes=15)])
        new_row_time = t + pd.DateOffset(minutes=15)
        # Create a new row with the same columns as df_SOC_
        new_row = pd.Series(index=df_SOC_.columns)
        # Get the current SOC values
        #SOC_t = df_SOC_.loc[t-pd.DateOffset(minutes=15)].to_dict()
    else:
        # If SOC_dict is empty or t-1 does not exist, use the initial df_SOC
        df_SOC_ = df_SOC
        SOC_t = df_SOC.shift().loc[t].to_dict()
        new_row_time = t + pd.DateOffset(minutes=15)
        # Create a new row with the same columns as df_SOC_
        new_row = pd.Series(index=df_SOC_.columns)

    # we compute blocked parked critical out of df_SOC_

    parked_prev = df_cars_month.shift().loc[t] == 1
    driving_prev = ~parked_prev
    critical = ((df_SOC_.loc[t -pd.DateOffset(minutes=15), parked_prev] - (discharging_rate * delta_t) / battery_capacity) <=
                df_SOC_min_month.loc[t, parked_prev]) & (df_cars.loc[t, parked_prev] == 0)
    #charge critical
    critical_parked_indices = df_SOC_.loc[t - pd.DateOffset(minutes=15), critical & parked_prev].index

    new_row[critical & parked_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), critical & parked_prev] + (
                charging_rate * delta_t) / battery_capacity
    ##need to replicate : new_row[critical & parked_prev]
    #df_Power.loc[t, critical & parked_prev] = -(charging_rate * scaling_factor)/1000
    Pagg_critical = (critical & parked_prev).sum()*(-charging_rate * scaling_factor)/1000
    Pdifference_left = df_15.shift().loc[t, 'Pdifference'] + Pagg_critical
    #df_CarCounts.loc[t, 'Pdifference_left'] = Pdifference_left

    ######
    if Pdifference_left > 0:
        condition2 = df_SOC_.loc[t -pd.DateOffset(minutes=15), parked_prev & ~critical] <= 1 - (charging_rate * delta_t) / battery_capacity
        # update cars that cannot be charged because soc is too high--> called the blocked cars
        new_row[~condition2 & ~critical& parked_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), ~condition2 & ~critical& parked_prev]
        #df_Power.loc[t, ~condition2 & ~critical& parked_prev] =0
        relevant_columns=charge(df_SOC_,charging_rate,scaling_factor,delta_t,battery_capacity,df_Power,df_CarCounts)
    if Pdifference_left < 0:
        condition = df_SOC_.loc[t -pd.DateOffset(minutes=15), parked_prev & ~critical] > (discharging_rate * delta_t) / battery_capacity
        new_row[~condition & ~critical & parked_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), ~condition2 & ~critical& parked_prev]
        #df_Power.loc[t, ~condition& ~critical&parked_prev ] =0
        relevant_columns=discharge(df_SOC_,discharging_rate,scaling_factor,delta_t,battery_capacity,df_Power,df_CarCounts)
    relevant_columns_indices = np.concatenate((relevant_columns, critical_parked_indices))
    # Add the new row to df_SOC_
    df_SOC_.loc[new_row_time] = new_row
    #replicate the mobility patterns (for all N we have same driving and soc min)

    #df_SOC_min_month[relevant_columns_indices]
    # need to replicate new_row[new_Power.index[:Nb]]

    # Generate the SOC list for this time step ###these are only the ones i know exist mutliple times because of scaling.
    SOC_list_t = get_SOC_list_t(df_SOC_, relevant_columns_indices, t - pd.DateOffset(minutes=15), scaling_factor)
    df_cars_month_ext, column_mapping = get_replicate_df(df_cars_month, relevant_columns_indices, scaling_factor)
    df_SOC_min_month_ext, column_mapping = get_replicate_df(df_SOC_min_month, relevant_columns_indices, scaling_factor)

    # Create a new index list by mapping the old column names to the list of repeated new column names
    new_index = [column_mapping[col_name][i % scaling_factor] for i, col_name in enumerate(SOC_list_t.index)]
    # Update the index of SOC_list_t with the new index list
    SOC_list_t.index = new_index
    SOC_dict_t = SOC_list_t.to_dict()


    df_SOC_min_month = df_SOC_min_month.join(df_SOC_min_month_ext)
    df_cars_month = df_cars_month.join(df_cars_month_ext)

    df_SOC_.loc[new_row_time] = new_row
    SOC_t = df_SOC_.loc[new_row_time].to_dict()
    SOC_dict[t] = {**SOC_t, **SOC_dict_t}
    # do we need to compute df_power expliciteley ?
    #df_Power = pd.DataFrame(0, index=df_cars_month.index, columns=df_cars_month.columns)


