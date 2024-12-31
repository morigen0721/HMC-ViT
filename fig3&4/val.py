## Use methods from the compare module to compare performance on test data
## Specify the method to use in global variables
## Process the generated test data and save the results at a specified location for visualization
## The generated result files are in CSV format, containing experimental conditions and RMSE
## Xie Yuxuan 2024-08-22 UTF-8
import os
import numpy as np
from toolbox import *
import concurrent.futures
import importlib
import time
from multiprocessing import Value, Lock
import pandas as pd
from plot import *

data_dir = '/home/yxxie/Documents/mutli_cls_tran_cnn/fig3_performance/val_data_2'  # Path to test data
num_cpus = 1
files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]  # Get a list of all .npz files
compare = importlib.import_module('compare')  # Dynamically import the compare module to call its functions dynamically
result_dir = '/home/yxxie/Documents/mutli_cls_tran_cnn/fig3_performance/val_result_2'  # Path to save results
method_name_list = ['Music'] 
method_type = 'sdsas'  # Specify method type, 'DL' for deep learning methods, others for traditional methods

# Define a shared counter for multi-process synchronization to show progress in real time
counter = Value('i', 0)
lock = Lock()  # Lock to ensure thread-safe counter operations

## Processing function: read .npz files and compute results
def process_file(file):

    global counter  # Use the global counter for multi-process synchronization
    start_time = time.time()  # Record start time

    meta_data = np.load(os.path.join(data_dir, file))  # Read the .npz file and extract metadata containing experimental conditions and covariance matrices
    method = getattr(compare, method_name)  # Get the method from the compare module

    # Convert data conditions to non-numpy arrays for pandas compatibility
    data = meta_data['data_cov']  # Extract covariance matrix data
    sig_num = meta_data['sig_num'].astype(int).item()  # Extract the number of sources
    doa = tuple(meta_data['doa'])  # Extract the actual DOA
    snr = meta_data['snr'].item()  # Extract the signal-to-noise ratio
    snap_num = meta_data['snap_num'].item()  # Extract the number of snapshots
    RMSE = 0  # Initialize RMSE

    ## If the method being evaluated is a traditional method, process each Monte Carlo experiment and compute RMSE
    if method_type != 'DL':
        ## Compute results for all Monte Carlo experiments
        for i in range(data.shape[0]):
            mc_data = data[i,:,:]  # Extract data for one Monte Carlo experiment
            est_doa = method(mc_data, 16, sig_num, 0.3, 1)  # Call the compare module method to estimate DOA
            RMSE += get_MSE(est_doa, meta_data['doa'])  # Calculate and accumulate RMSE for this experiment
    
    ## If the method being evaluated is a deep learning method, use GPU for acceleration
    ## Process all Monte Carlo experiments under the same conditions in a single matrix operation
    if method_type == 'DL':
        RMSE = method(data, sig_num, doa)  # Call the compare module method to estimate DOA

    ## Assemble experimental results for this condition
    result = {
        'RMSE': np.sqrt(RMSE/data.shape[0]).item(),
        'sig_num': sig_num,
        'doa': doa,
        'snr': snr,
        'snap_num': snap_num
    }

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate execution time
    # Update the counter
    with lock:
        counter.value += 1
        print(f"Processed {counter.value} conditions, took {elapsed_time:.4f} seconds for {data.shape[0]} Monte Carlo experiments under this condition.")

    return result

## Main function
if __name__ == '__main__':

    for i in range(len(method_name_list)):

        method_name = method_name_list[i]

        start_time = time.time()  # Record start time

        ## Save experimental results
        folder_path = os.path.join(result_dir, method_name)
        csv_file_path = os.path.join(folder_path, 'results.csv')

        ## Check if experimental results have already been computed
        if not os.path.exists(csv_file_path):  # If not, call the process_file function to compute results in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
                results = list(executor.map(process_file, files))
            os.makedirs(folder_path, exist_ok=True)
            # Organize experimental results into a DataFrame
            df = pd.DataFrame(results)
            df = df.groupby(['sig_num', 'snr', 'snap_num'])['RMSE'].mean().reset_index()  # Average data with the same sig_num for different DOAs
            df.to_csv(csv_file_path, index=False)  # Save experimental results to a CSV file
        else:  # If results exist, read them directly
            print(f"Results for experiment '{method_name}' already exist.")
            df = pd.read_csv(csv_file_path)

        end_time = time.time()  # Record end time
        print(f"Total testing time for experiment '{method_name}' was {end_time - start_time:.4f} seconds.")
        plot_results_from_folders(result_dir)  # Plot experimental results