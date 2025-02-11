import torch
import numpy as np
import os



import torch
import numpy as np
import os
import h5py


# Get file size in MB
file_size_MB = os.path.getsize('data/SH3D_600_Nt_101_Nx_80.mat') / (1024 * 1024)

print(f"File successfully converted! Size: {file_size_MB:.2f} MB")

# Open the .mat file
with h5py.File('data/SH3D_600_Nt_11_Nx_80.mat', 'r') as f:
    phi = f['phi']  # Access the dataset

    print("Shape of phi:", phi.shape)
    print("Data type:", phi.dtype)

    # Print a small subset of the data
    sample_data = phi[0, 0, 0, :5, :5]  # Extract a small portion
    print("Sample values:\n", sample_data)
    # Print attributes (if any)
    print("Attributes of phi:", dict(phi.attrs))
# Get file size in MB
file_size_MB = os.path.getsize('data/CH3D_2000_Nt_101_Nx_32.mat') / (1024 * 1024)
print(f"Compressed file size: {file_size_MB:.2f} MB")

# Open the .mat file
with h5py.File('data/CH3D_2000_Ntt_101_Nx_32.mat', 'r') as f:
    phi = f['phi']  # Access the dataset

    print("Shape of phi:", phi.shape)





    # Extract first, second, and last arrays along the last axis
    first_array = phi[..., 0]  # First array at index 0
    second_array = phi[..., 1]  # Second array at index 1
    last_array = phi[..., -1]  # Last array at index -1

    print("\nFirst array:\n", first_array)
    print("\nSecond array:\n", second_array)
    print("\nLast array:\n", last_array)

# Load the .pt file
data = torch.load('/scratch/noqu8762/PhaseField1/data/CH3D_8000_Nt_101_Nx_32.pt')

# Extract the tensor from the dictionary
tensor = data['data']  # Access the tensor inside the 'data' key

# Convert the tensor to a NumPy array
np_array = tensor.cpu().numpy()  # Move to CPU if needed and convert

# Save the NumPy array to an .npz file
np.savez('CH3D_8000_Nt_101_Nx_32.npz', data=np_array)


'''
import h5py

h5_save_path = "/scratch/noqu8762/PhaseField1/data/CH3D_8000_Nt_101_Nx_32.h5"

# Load the HDF5 file
with h5py.File(h5_save_path, 'r') as h5_file:
    keys = list(h5_file.keys())  # Get variable names
    print("Variables in HDF5 file:", keys)

    # Select the first variable (change if needed)
    variable_name = keys[0]
    data = h5_file[variable_name][:]  # Read the dataset

    # Print first and last 5 rows
    print(f"First 5 rows of {variable_name}:\n", data[:5])
    print(f"Last 5 rows of {variable_name}:\n", data[-5:])
'''