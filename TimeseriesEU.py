import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


x_intersections=[2.87787788,6.97597598,11.05105105,13.83683684,14.27427427]
diff=[1,-2]
x_intersections.insert(0, 0)
x_intersections.append(24)
if diff[0]>0:
    array1 = [[x_intersections[i], x_intersections[i+1]] for i in range(0, len(x_intersections)-1, 2)]
    array2 = [[x_intersections[i], x_intersections[i+1]] for i in range(1, len(x_intersections)-1, 2)]
else:
    array2 = [[x_intersections[i], x_intersections[i+1]] for i in range(0, len(x_intersections)-1, 2)]
    array1 = [[x_intersections[i], x_intersections[i+1]] for i in range(1, len(x_intersections)-1, 2)]

print("Array 1:", array1)
print("Array 2:", array2)


