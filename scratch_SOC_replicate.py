import pandas as pd
import numpy as np
import math
import random
import hashlib
import datetime


month=8
capacity=9
scaledown=200
df_cars = pd.read_csv('expanded_dataset.csv', index_col='time_interval')
df_SOC_min = pd.read_csv('SOCmin_data2.csv' , index_col='time_interval',parse_dates=True)
#df15 = pd.read_csv(f'Wind_NY_Power{capacity}', index_col=0, parse_dates=True)
df15 = pd.read_csv('Wind_NY_Power9.csv', index_col=0, parse_dates=True)
df15.index = pd.to_datetime(df15.index)
df_cars.index = pd.to_datetime(df_cars.index)
df_cars = df_cars.sort_values(by='time_interval')


charging_rate=50
discharging_rate=20
delta_t=0.25
battery_capacity=70
energy_loss_driving=0.75


# function declaration
def load_and_extract_month(df_cars,month):
    df_cars.index = pd.to_datetime(df_cars.index)
    df_cars = df_cars.sort_values(by='time_interval')
    df_cars = df_cars[df_cars.index.month == month]
    df_cars = df_cars.drop('total', axis=1)
    return df_cars


def get_SOC_list_t(df_SOC, relevant_columns_indices, t, scaling_factor):
    # Get the relevant columns from df_SOC
    df_relevant = df_SOC.loc[t, relevant_columns_indices]

    # Get the SOC values you want to replicate
    SOC_values_to_replicate = df_relevant.values

    # Get the indices to replicate
    indices_to_replicate = df_relevant.index

    # Repeat the SOC values once
    replicated_SOC_values = np.repeat(SOC_values_to_replicate, 1)

    ## we assumed that we have each car Nb, alpha times with the exact same mobility patterns. However we don't assume that the SOC of these replicated cars ar ein similar states.
    # They need to be in available state and a better assumption is to stay that when they have high SOC they to discharge and when the have low SOC the want to discharge.
    # The SOC value at t+1 can then be initialized randomly as follows:
    # Generate random values (scaling_factor - 1) times
    if Pdifference_left>0:
        random_SOC_values = np.random.uniform(0.2, 0.5, size=(len(SOC_values_to_replicate), scaling_factor - 1))
    if Pdifference_left<0:
        random_SOC_values = np.random.uniform(0.6, 1, size=(len(SOC_values_to_replicate), scaling_factor - 1))

    # Concatenate the replicated values and the random values
    final_SOC_values = np.hstack((replicated_SOC_values[:, np.newaxis], random_SOC_values)).flatten()

    # Repeat the indices scaling_factor times
    replicated_indices = np.repeat(indices_to_replicate, scaling_factor)

    # Return a Series with final SOC values and replicated indices
    return pd.Series(final_SOC_values, index=replicated_indices)


def get_replicate_df(df_cars_month, relevant_column_indices, scaling_factor, j, t, column_mapping):
    # Get the rows corresponding to t
    df_specific_times = df_cars_month.loc[[t]]
    # Repeat the relevant columns using NumPy broadcasting
    replicated_cols = np.tile(df_specific_times[relevant_column_indices].values, (1, scaling_factor))
    # Check if we have a pre-existing column mapping
    if column_mapping is None:
        # Create new column names for the replicated columns
        new_col_names = []
        column_mapping = {col_name: [] for col_name in relevant_column_indices}
        for col_name in relevant_column_indices:
            for i in range(scaling_factor):
                random_id = random.randint(1, 10**6)  # Reduce the upper limit to decrease the length of the ID
                unique_id = hashlib.md5(f"{j}{i}{random_id}".encode()).hexdigest()[-7:]  # Get last 7 characters of the hash
                new_name = f"{j}{i}{unique_id}{col_name}"
                new_col_names.append(new_name)
                column_mapping[col_name].append(new_name)
    else:
        # If we have a pre-existing column mapping, use it to generate new column names
        new_col_names = [new_name for col_name in relevant_column_indices for new_name in column_mapping[col_name]]

    # Create the replicated DataFrame using the replicated columns and new column names
    replicated_df = pd.DataFrame(replicated_cols, index=df_specific_times.index, columns=new_col_names)

    return replicated_df, column_mapping


def get_related_dataframes(df_cars_month_joined,df_SOC_min_month_joined,df_cars_month, df_SOC_min_month, t):
    df_cars_month_all= df_cars_month_joined.loc[[t]]
    df_SOC_min_month_all = df_SOC_min_month_joined.loc[[t]]
    indices=df_cars_month_joined.loc[[t]].columns
    original_indices = [index[-7:] for index in indices]
    df_cars_data_t_plus_1 = df_cars_month.loc[
       t + pd.DateOffset(minutes=15), original_indices]
    df_SOC_min_data_t_plus_1 = df_SOC_min_month.loc[
       t + pd.DateOffset(minutes=15), original_indices]

     #Create a new dataframe with the data at t+1 and the corresponding new column names
    df_cars_month_t_plus_1 = pd.DataFrame([df_cars_data_t_plus_1.values],
                                                      index=[t + pd.DateOffset(minutes=15)],
                                                      columns=indices)
    df_SOC_min_data_t_plus_1 = pd.DataFrame([df_SOC_min_data_t_plus_1.values],
                                                        index=[t + pd.DateOffset(minutes=15)],
                                                        columns=indices)
    df_cars_month_joined = pd.concat([df_cars_month_all, df_cars_month_t_plus_1])
    df_SOC_min_month_joined = pd.concat([df_SOC_min_month_all, df_SOC_min_data_t_plus_1])

    return df_cars_month_joined, df_SOC_min_month_joined


def get_related_dataframes_non_relevant(df_cars_month_ext_non_relevant,df_SOC_min_month_ext_non_relevant,df_cars_month, df_SOC_min_month, t, non_relevant_columns_indices):
    # Considering non_relevant_columns_indices as the original indices for non-relevant columns
    original_indices_non_relevant = [index[-7:] for index in non_relevant_columns_indices]

    df_cars_data_t_plus_1_non_relevant = df_cars_month.loc[
        t + pd.DateOffset(minutes=15), original_indices_non_relevant]
    df_SOC_min_data_t_plus_1_non_relevant = df_SOC_min_month.loc[
        t + pd.DateOffset(minutes=15), original_indices_non_relevant]

    # Create a new dataframe with the data at t+1 and the corresponding new column names
    df_cars_month_t_plus_1_non_relevant = pd.DataFrame([df_cars_data_t_plus_1_non_relevant.values],
                                                       index=[t + pd.DateOffset(minutes=15)],
                                                       columns=non_relevant_columns_indices)
    df_SOC_min_data_t_plus_1_non_relevant = pd.DataFrame([df_SOC_min_data_t_plus_1_non_relevant.values],
                                                         index=[t + pd.DateOffset(minutes=15)],
                                                         columns=non_relevant_columns_indices)

    df_cars_month_not_rel = pd.concat([df_cars_month_ext_non_relevant, df_cars_month_t_plus_1_non_relevant])
    df_SOC_min_month_not_rel = pd.concat([df_SOC_min_month_ext_non_relevant, df_SOC_min_data_t_plus_1_non_relevant])

    return df_cars_month_not_rel, df_SOC_min_month_not_rel


def get_related_dataframes_relevant(df_cars_month, df_SOC_min_month, df_cars_month_ext, df_SOC_min_month_ext, t, column_mapping):
    # Get the new column names from the column mapping
    new_column_names = [name for names in column_mapping.values() for name in names]

    # Create inverse column mapping using dictionary comprehension and string slicing
    master_inv_column_mapping = {new_column: new_column[-7:] for new_column in new_column_names}

    df_cars_data_t_plus_1 = df_cars_month.loc[t + pd.DateOffset(minutes=15), master_inv_column_mapping.values()]
    df_SOC_min_data_t_plus_1 = df_SOC_min_month.loc[t + pd.DateOffset(minutes=15), master_inv_column_mapping.values()]

    # Create new dataframes with the data at t+1 and the corresponding new column names
    df_cars_data_t_plus_1 = pd.DataFrame([df_cars_data_t_plus_1.values], index=[t + pd.DateOffset(minutes=15)],
                                         columns=master_inv_column_mapping.keys())
    df_SOC_min_data_t_plus_1 = pd.DataFrame([df_SOC_min_data_t_plus_1.values], index=[t + pd.DateOffset(minutes=15)],
                                            columns=master_inv_column_mapping.keys())

    # Concatenate df_cars_month_ext and df_SOC_min_month_ext with the new df_t_plus_1 dataframes
    df_cars_month_rel = pd.concat([df_cars_month_ext, df_cars_data_t_plus_1])
    df_SOC_min_month_rel = pd.concat([df_SOC_min_month_ext, df_SOC_min_data_t_plus_1])

    return df_cars_month_rel, df_SOC_min_month_rel


def charge(df_SOC_,Pdifference_left,rate,delta_t,battery_capacity):
    # Sort the SOC by lowest to highest from the available ones excluding the critical cars
    SOC_sorted = df_SOC_.loc[t - pd.DateOffset(minutes=15), condition2 & ~critical & parked_prev].sort_values(
        ascending=True)
    # 3. The replication of alpha cars means that we will also have alpha times a car with similar mobility patterns ans SOC in available state.
    # 4. Decide if remaining cars need to be charged
    # check if we need to remain charging
    if (Pdifference_left-(charging_rate) / 1000)<0: #this means we don't charge anymore cars.
        new_Power = pd.Series(0, index=SOC_sorted.index)
        Nb = 0  # we don't charge anymore
        relevant_columns = new_Power.index[:Nb]
        new_row[new_Power.index[:Nb]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[:Nb]]
        new_row[new_Power.index[Nb:]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[Nb:]]
        PV2G_charge = 0 + Pagg_critical
        free_nb=len(SOC_sorted)-Nb
        alpha=1
    else:
        needed= math.ceil(abs(Pdifference_left) * 1000 / (rate))-1
        # 2. alpha: Define how many cars need to be replicated
        if needed > len(SOC_sorted):
            alpha = math.ceil(needed / len(SOC_sorted))
            Nb = math.ceil(needed / alpha) - 1
        else:
            alpha = 1
            Nb=needed
        if len(SOC_sorted)==1:
            Nb=1

        new_Power = pd.Series((-rate * alpha) / 1000, index=SOC_sorted.index)

        new_row[new_Power.index[:Nb]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[:Nb]] + (
                    rate * delta_t) / battery_capacity
        relevant_columns = new_Power.index[:Nb]

        new_row[new_Power.index[Nb:]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[Nb:]]

        PV2G_charge=-Nb*alpha*(rate)/1000+ Pagg_critical
        free_nb = len(SOC_sorted) - Nb
    return Pdifference_left,relevant_columns,alpha,Nb,PV2G_charge,free_nb

def discharge(df_SOC_,Pdifference_left,rate,delta_t,battery_capacity):
    print('discharging')
    SOC_sorted = df_SOC_.loc[t - pd.DateOffset(minutes=15), condition & ~critical & parked_prev].sort_values(ascending=False)
    print('available discharging', len(SOC_sorted))
    needed = math.ceil(abs(Pdifference_left) * 1000 / (rate))
    if needed> len(SOC_sorted):
        alpha=math.ceil(needed/len(SOC_sorted))
        Nb=math.ceil(needed/alpha)-1
        if len(SOC_sorted) == 1:
            Nb = 1
    else:
        alpha=1
        Nb= needed
    # Calculate new Power for all parked, not critical cars
    # power given in KW we want it in MW need to dived by 1000 to get MW
    new_Power = pd.Series((rate * alpha) / 1000, index=SOC_sorted.index)

    new_row[new_Power.index[:Nb]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[:Nb]] - (
            rate * delta_t) / battery_capacity
    relevant_columns = new_Power.index[:Nb]
    new_row[new_Power.index[Nb:]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    PV2G_discharge=Nb*alpha*(rate)/1000
    free_nb=len(SOC_sorted)-Nb

    return relevant_columns,alpha,Nb,PV2G_discharge,free_nb


def discharge_after_charge(df_SOC_,Pdifference_left_new,rate,alpha,delta_t, battery_capacity):
    # available_discharge = (df_SOC.shift().loc[t,parked_prev] != 0 & ~critical)
    print('discharge after critical')
    SOC_sorted = df_SOC_.loc[t - pd.DateOffset(minutes=15), condition & ~critical & parked_prev].sort_values(
        ascending=False)
    print('available discharging',len(SOC_sorted))
    needed= math.ceil(abs(Pdifference_left_new) * 1000 / (rate))
    if needed > len(SOC_sorted):
        print('fail algorithm')
    if alpha>1:
        Nb=math.ceil(needed/alpha)-1
    else:
        Nb= needed
    if len(SOC_sorted)==1:
        Nb=1
    new_Power = pd.Series((rate * alpha) / 1000, index=SOC_sorted.index)
    # Determine the number of cars for which we need to calculate the Power
    new_row[new_Power.index[:Nb]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[:Nb]] - (
            rate * delta_t) / battery_capacity
    relevant_columns = new_Power.index[:Nb]

    new_row[new_Power.index[Nb:]] = df_SOC_.loc[t - pd.DateOffset(minutes=15), new_Power.index[Nb:]]
    # Update df_Power for car_ids in 'result'
    PV2G_discharge = Nb * alpha * (rate) / 1000
    free_nb = len(SOC_sorted) - Nb
    # df_Power.loc[t, new_Power.index[Nb:]] = 0
    return relevant_columns,Nb,PV2G_discharge,free_nb



### main

df_cars_month =load_and_extract_month(df_cars ,month)
df_SOC_min_month = load_and_extract_month(df_SOC_min, month)
df_15 = df15[df15.index.month == month]/scaledown

first_index = df_cars_month.index[0]
# Initialize randomly the SOC for one row
df_SOC = pd.DataFrame(
    np.random.choice(np.linspace(0.1, 1, 10), size=(1, df_cars_month.shape[1])),
    index=[first_index],
    columns=df_cars_month.columns
)

SOC_dict = {}
data_dicts = []
j=0
master_column_mapping = {}
# Initialize an empty dictionary to hold the master inverse column mapping
master_inv_column_mapping = {}
column_mapping = None
for t in df_cars_month.index[1:]:
    print(t)
    # Check if the timestamp exists in SOC_dict
    # If SOC_dict is not empty and t-1 exists in SOC_dict, convert the last entry to a dataframe
    if SOC_dict and (t - pd.DateOffset(minutes=15)) in SOC_dict:
        # Convert your datetime t-1 to a dataframe
        df_SOC_ = pd.DataFrame(SOC_dict[t - pd.DateOffset(minutes=15)], index=[t - pd.DateOffset(minutes=15)])
        new_row_time = t
        # Create a new row with the same columns as df_SOC_
        new_row = pd.Series(index=df_SOC_.columns)
    else:
        # If SOC_dict is empty or t-1 does not exist, use the initial df_SOC
        df_SOC_ = df_SOC
        new_row_time = t
        new_row = pd.Series(index=df_SOC_.columns)
        df_cars_month_joined=df_cars_month
        df_SOC_min_month_joined=df_SOC_min_month

    ### create mask for which the different states of a car can be taken
    parked_prev = df_cars_month_joined.shift().loc[t] == 1
    driving_prev = ~parked_prev
    #critical cars
    critical = ((df_SOC_.loc[t -pd.DateOffset(minutes=15), parked_prev] - (discharging_rate * delta_t) / battery_capacity) <=df_SOC_min_month_joined.loc[t,parked_prev]) & (df_cars_month_joined.loc[t, parked_prev] == 0)
    critical_parked_indices = df_SOC_.loc[t - pd.DateOffset(minutes=15), critical& parked_prev].index
    # blocked cars for either charging or discharging
    condition2 = df_SOC_.loc[t - pd.DateOffset(minutes=15), parked_prev & ~critical] <= 1 - (charging_rate * delta_t) / battery_capacity
    condition = df_SOC_.loc[t - pd.DateOffset(minutes=15), parked_prev & ~critical] > (discharging_rate * delta_t) / battery_capacity
    ### update the SOC of all driving cars at time step t+1
    new_row[driving_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), driving_prev] - energy_loss_driving / battery_capacity
    new_row[critical & parked_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), critical & parked_prev] + (charging_rate * delta_t) / battery_capacity
    # check for periods of high wind or low wind.
    Pdifference = df_15.shift().loc[t, 'Pdifference']
    Pagg_critical = (critical & parked_prev).sum() * (-charging_rate) / 1000
    Pdifference_left= Pdifference+Pagg_critical
    print(Pdifference_left)
    if Pdifference_left > 0:
        # compute SOC of blocked cars (cannot be charged) SOC(t+1)=SOC(t)
        new_row[~condition2 & ~critical & parked_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), ~condition2 & ~critical & parked_prev]
        # charge cars.
        Pdifference_left_new,relevant_columns,alpha,Nb,PV2G,free_nb=charge(df_SOC_,Pdifference_left,charging_rate,delta_t,battery_capacity)
        #relevant_columns_indices = np.concatenate((relevant_columns, critical_parked_indices))
        relevant_columns_indices = relevant_columns

    if Pdifference_left < 0:
        new_row[~condition & ~critical & parked_prev] = df_SOC_.loc[t - pd.DateOffset(minutes=15), ~condition & ~critical& parked_prev]
        relevant_columns,alpha,Nb,PV2G,free_nb=discharge(df_SOC_,Pdifference_left,discharging_rate,delta_t,battery_capacity)
        #relevant_columns_indices = np.concatenate((relevant_columns, critical_parked_indices))
        relevant_columns_indices = relevant_columns


    relevant_columns_indices_set = set(relevant_columns_indices)

    # these are all the ones that or either driving, or  blocked by condition 1 or 2 or  free --> not participating because power balance is fulfilled
    non_relevant_columns_indices = [idx for idx in df_SOC_.columns if idx not in relevant_columns_indices_set]

    if alpha>1: ## adding new cars with similar mobility patterns and SOC in good range
        df_SOC_.loc[new_row_time] = new_row
        # we need this because we need to also store the SOC of the not relevant column indices ( those that are not scaled)

        # Generate the SOC list for the next  time step ###these are only the ones i know exist mutliple times because of scaling.
        SOC_list_t = get_SOC_list_t(df_SOC_, relevant_columns_indices, t , alpha)

        # create a dictionary with the soc of all the non relavent cars ( driving, blocked, free)
        SOC_t_not_relevant = df_SOC_.loc[t, non_relevant_columns_indices].to_dict()

        df_cars_month_ext_non_relevant = df_cars_month_joined.loc[[t], non_relevant_columns_indices]
        df_SOC_min_month_ext_non_relevant = df_SOC_min_month_joined.loc[[t], non_relevant_columns_indices]

        # we need this for the sake of taking into account upcoming drives, meaning we need the correct SOCmin.
        # if it weren't for SOC min and just recreating mobility pattens we could just compute the probability of driving/parking at time t
        df_cars_month_not_rel, df_SOC_min_month_not_rel = get_related_dataframes_non_relevant(df_cars_month_ext_non_relevant,df_SOC_min_month_ext_non_relevant,df_cars_month, df_SOC_min_month, t, non_relevant_columns_indices)

        df_cars_month_ext,column_mapping = get_replicate_df(df_cars_month_joined, relevant_columns_indices, alpha,j,t,column_mapping)
        df_SOC_min_month_ext,column_mapping = get_replicate_df(df_SOC_min_month_joined, relevant_columns_indices, alpha,j,t,column_mapping)

        # Update the index of SOC_list_t with the new index list to get the corresponding index of the replicated cars.
        new_index = [column_mapping[col_name][i % (alpha)] for i, col_name in enumerate(SOC_list_t.index)]
        SOC_list_t.index = new_index
        SOC_dict_t = SOC_list_t.to_dict()

        # Update master column mapping to keep track of correct indices
        master_column_mapping.update(column_mapping)

        df_cars_month_rel, df_SOC_min_month_rel = get_related_dataframes_relevant(df_cars_month, df_SOC_min_month,df_cars_month_ext, df_SOC_min_month_ext, t,column_mapping)

        df_SOC_min_month_joined=df_SOC_min_month_not_rel.join(df_SOC_min_month_rel)
        df_cars_month_joined=df_cars_month_not_rel.join(df_cars_month_rel)

        SOC_dict[t] = {**SOC_t_not_relevant,**SOC_dict_t} # this dictionary now has all the car ids and SOC status of all cars at t+1.
        column_mapping = None

    else: ### we don't need replication in as we have enough cars but we do need to get df_cars_month and dfSOC_joined at time t+1
        df_SOC_.loc[new_row_time] = new_row
        SOC_t = df_SOC_.loc[new_row_time].to_dict()
        SOC_dict[t] = {**SOC_t} # this dictionary now has all the car ids and SOC status of all cars at t+1.

        df_cars_month_joined, df_SOC_min_month_joined = get_related_dataframes(df_cars_month_joined, df_SOC_min_month_joined,df_cars_month,df_SOC_min_month, t)
        column_mapping = None


    data_dict = {
        'index t': t-pd.DateOffset(minutes=15),  # Now t is defined as your loop variable
        'Size': df_SOC_.shape[1],
        'parked ': parked_prev.sum(),
        'alpha ': alpha,  # Now alpha should be defined inside your loop
        'Blocked discharging ': (~condition & ~critical & parked_prev).sum(),
        'Blocked charging ': (~condition2 & ~critical & parked_prev).sum(),
        'critical ': (critical).sum(),
        'driving ': (driving_prev).sum(),
        'Nb': Nb,
        'free cars':free_nb,
        'TotalParticipation needed':df_cars_month_joined.shape[1]-free_nb,
        'P_difference_left': Pdifference_left,  # Now P_difference_left should be defined inside your loop
        'PV2G': PV2G,  # Now PV2G should be defined inside your loop

    }
    # Append the data dictionary to the list
    data_dicts.append(data_dict)
    #print(data_dicts)

# Convert the list of dictionaries to a pandas DataFrame
df_to_export = pd.DataFrame(data_dicts)

# Export the DataFrame to a CSV file
df_to_export.to_csv(f'scaledown_{scaledown}Month{month}_rate_c{charging_rate}_rate_d{discharging_rate}_WIND{capacity}000MW.csv', index=False)



