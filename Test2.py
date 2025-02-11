
import torch.nn as nn
import torch.nn.functional as F

print(bin(352))

layer = nn.Linear(10, 5)
print(layer)

def compute_loss(values, size_average=True):
    total_loss = sum(values)

    if size_average:  # If True, return the average
        return total_loss / len(values)
    else:  # If False, return the sum
        return total_loss

# Example usage
numbers = [2, 4, 6, 8, 10]

print(compute_loss(numbers))          # Default: size_average=True, returns average (6.0)
print(compute_loss(numbers, True))    # Explicitly using True, also returns average (6.0)
print(compute_loss(numbers, False))   # Using False, returns sum (30)

import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'Name': ['A', 'B', 'C', 'D', 'E'],
    'Price': [100, 200, 150, 250, 180],
    'Size': [50, 60, 55, 65, 58],
    'Year': [2020, 2018, 2019, 2021, 2017]
})

print(df.max(axis=0))  # Max for each column (axis=0, column-wise)
'''
axis=0 (column-wise): Finds the max for each column.
'''

print(df.max(axis=1))  # Max for each row (axis=1, row-wise)