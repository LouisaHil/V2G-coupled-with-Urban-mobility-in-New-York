import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# simulate 10 hours of data, sampled every 15 minutes
minutes_in_10_hours = 24*60
time = np.linspace(0, minutes_in_10_hours, minutes_in_10_hours // 15)

# initial state of charge
state_of_charge = np.full_like(time, 0.5)

# create areas for high wind, low wind, and driving periods
# since there's always either high wind or low wind, we'll alternate between them
high_wind = np.array([True if (i // 80) % 2 == 0 else False for i in range(len(time))])
low_wind = ~high_wind

# Let's say driving happens in the middle of each high wind period and lasts for 45 minutes
driving = np.array([True if i % 80 >= 27 and i % 80 < 27 + 18 else False for i in range(len(time))])

# rate of SOC increase/decrease every 15 minutes
rate = 0.178

# modify SOC based on conditions
for i in range(1, len(time)):
    if high_wind[i] and not driving[i]:
        state_of_charge[i] = min(1.0, state_of_charge[i-1] + rate)
    elif driving[i]:
        state_of_charge[i] = max(0.1, state_of_charge[i - 1] - 0.02)
    else:
        state_of_charge[i] = max(0.1, state_of_charge[i-1] - rate)

# plot the state of charge
plt.figure(figsize=(12, 6))
plt.plot(time, state_of_charge, label='State of Charge')
plt.xlabel('Time (minutes)')
plt.ylabel('State of Charge')

# highlight the different periods
plt.fill_between(time, 0, 1, where=high_wind, color='green', alpha=0.3, label='High Wind')
plt.fill_between(time, 0, 1, where=low_wind, color='red', alpha=0.3, label='Low Wind')
plt.fill_between(time, 0, 1, where=driving, color='blue', alpha=0.3, label='Driving Period')

# legend
green_patch = mpatches.Patch(color='green', alpha=0.3, label='High Wind')
red_patch = mpatches.Patch(color='red', alpha=0.3, label='Low Wind')
blue_patch = mpatches.Patch(color='blue', alpha=0.3, label='Driving Period')
plt.legend(handles=[green_patch, red_patch, blue_patch])

plt.title('State of Charge behavior')
plt.grid(True)
plt.show()