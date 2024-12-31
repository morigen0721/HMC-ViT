## Related methods for comparison, including traditional methods and replicated deep learning-based methods
## Aim to consistently accept covariance matrices as input and output estimated DOA
## 2024-08-22 Xie Yuxuan UTF-8
import numpy as np
from dualtran import *
from cnnlow import *
from prop import *
import torch
from toolbox import *
import cvxpy as cp
from scipy.linalg import svd
from scipy.stats import chi2

## Deep learning-based method
## The model is defined in model.py in the same directory
## data - Covariance matrices from all Monte Carlo experiments; sig_num - Number of sources; real_doa - Actual DOA
## Model parameters are loaded from a specified path
def test_cnn(data, sig_num, real_doa):
    # Path to model parameters
    parameter_path = '/home/yxxie/Documents/mutli_cls_tran_cnn/fig3_performance/weights/exp-cnn/best_val.pth'

    # Data processing: 'data' is a stacked covariance matrix from 10000 Monte Carlo experiments, dimensions: 10000,16,16
    Re = np.real(data)
    Im = np.imag(data)
    Phase = np.angle(data)
    data = np.stack((Re, Im, Phase), axis=-1)  # Data dimensions: elements * elements * 3
    data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor
    data = data.permute(0, 3, 1, 2)  # Change from [height, width, channels] to [channels, height, width], ignoring batch size

    # Setup and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lowsnr_cnn()
    model.load_state_dict(torch.load(parameter_path, map_location=device))
    model.to(device)
    model.eval()
    data = data.to(device)  # Move input data to GPU

    # Perform predictions with the model
    with torch.no_grad():
        output = model(data)

    output = output.cpu().numpy()  # Move output data to CPU. Output dimensions: 10000,121
    RMSE = 0  # Initialize RMSE

    # Process the results of all Monte Carlo experiments
    for i in range(output.shape[0]):
        spectrum = output[i, :]  # Extract data for one Monte Carlo experiment
        est_doa = find_peaks(spectrum, sig_num) - (len(spectrum) - 1) / 2  # Estimate DOA using spatial spectrum method
        RMSE += get_MSE(est_doa, real_doa)  # Calculate and accumulate RMSE for this experiment
    
    return RMSE

def test_dualtran(data, sig_num, real_doa):
    # Path to model parameters
    parameter_path = '/home/yxxie/Documents/mutli_cls_tran_cnn/fig3_performance/weights/dualtran/best_val.pth'

    # Data processing: 'data' is a stacked covariance matrix from 10000 Monte Carlo experiments, dimensions: 10000,16,16
    Re = np.real(data)
    Im = np.imag(data)
    Phase = np.angle(data)
    data = np.stack((Re, Im, Phase), axis=-1)  # Data dimensions: elements * elements * 3
    data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor
    data = data.permute(0, 3, 1, 2)  # Change from [height, width, channels] to [channels, height, width], ignoring batch size

    # Setup and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dualtran()
    model.load_state_dict(torch.load(parameter_path, map_location=device))
    model.to(device)
    model.eval()
    data = data.to(device)  # Move input data to GPU

    # Perform predictions with the model
    with torch.no_grad():
        output = model(data)

    output = output.cpu().numpy()  # Move output data to CPU. Output dimensions: 10000,121
    RMSE = 0  # Initialize RMSE

    # Process the results of all Monte Carlo experiments
    for i in range(output.shape[0]):
        spectrum = output[i, :]  # Extract data for one Monte Carlo experiment
        est_doa = find_peaks(spectrum, sig_num) - (len(spectrum) - 1) / 2  # Estimate DOA using spatial spectrum method
        RMSE += get_MSE(est_doa, real_doa)  # Calculate and accumulate RMSE for this experiment
    
    return RMSE

def test_prop(data, sig_num, real_doa):
    # Path to model parameters
    parameter_path = '/home/yxxie/Documents/mutli_cls_tran_cnn/fig3_performance/weights/prop/best_val.pth'

    # Data processing: 'data' is a stacked covariance matrix from 10000 Monte Carlo experiments, dimensions: 10000,16,16
    Re = np.real(data)
    Im = np.imag(data)
    Phase = np.angle(data)
    data = np.stack((Re, Im, Phase), axis=-1)  # Data dimensions: elements * elements * 3
    data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor
    data = data.permute(0, 3, 1, 2)  # Change from [height, width, channels] to [channels, height, width], ignoring batch size

    sample = batch_extract_upper_triangle(data).view(data.shape[0], 3, 1, -1)

    # Setup and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prop()
    model.load_state_dict(torch.load(parameter_path, map_location=device))
    model.to(device)
    model.eval()
    sample = sample.to(device)  # Move input data to GPU

    # Perform predictions with the model
    with torch.no_grad():
        output = model(sample)

    output = output.cpu().numpy()  # Move output data to CPU. Output dimensions: 10000,121
    RMSE = 0  # Initialize RMSE

    # Process the results of all Monte Carlo experiments
    for i in range(output.shape[0]):
        spectrum = output[i, :]  # Extract data for one Monte Carlo experiment
        est_doa = find_peaks(spectrum, sig_num) - (len(spectrum) - 1) / 2  # Estimate DOA using spatial spectrum method
        RMSE += get_MSE(est_doa, real_doa)  # Calculate and accumulate RMSE for this experiment
    
    return RMSE

## Traditional MUSIC algorithm
## data - Covariance matrix; array_num - Number of array elements; sig_num - Number of sources; lamda - Wavelength; angle_gap - Angle interval
## Note: Ensure MUSIC is adjusted accordingly if angle range is -90 to 90 or 0 to 180
## Returns - Estimated DOA and unnormalized MUSIC spectrum
def Music(data, array_num, sig_num, lamda, angle_gap):
    # Initialization
    d = (np.arange(array_num) * lamda / 2).reshape(-1, 1)  # Array position vector
    grid_angle = np.arange(0, 180.0001, angle_gap)  # Angle grid, generating a grid of length 180/angle_gap + 1
    Pmusic = np.zeros(int(180 / angle_gap) + 1)  # Pmusic stores spatial spectrum values, same length as grid

    # Eigenvalue decomposition and sorting of eigenvalues and eigenvectors, then truncating eigenvectors
    EigValues, EigVec1 = np.linalg.eig(data)
    sorted_indices = np.argsort(EigValues)
    EigValues = EigValues[sorted_indices[::-1]]  # Sort eigenvalues in descending order
    EigVec1 = EigVec1[:, sorted_indices[::-1]]
    En = EigVec1[:, sig_num:]

    # Traverse all angles to reconstruct the spatial spectrum
    for i in range(len(grid_angle)):
        ##### Pay attention to handling angle range, sincos, angle adjustments (e.g., subtracting 90), etc.
        # Calculate the steering vector for the current angle iteration. Use sin since angles are in -90 to 90 range; adjust grid by subtracting 90
        A2 = np.exp(-1j * 2 * np.pi * d / lamda * np.sin(np.deg2rad(grid_angle[i] - 90)))
        # Multiply with noise subspace to determine orthogonality. Smaller values indicate stronger orthogonality
        W = np.matmul(A2.conj().T, np.matmul(np.matmul(En, En.conj().T), A2))
        # Check if denominator is zero
        if np.abs(W) > 1e-10:
            Pmusic[i] = abs(1 / W)
        else:
            print('Error: Issue with MUSIC algorithm. Some angles are almost perfectly orthogonal to noise subspace! This error is from the MUSIC function.')
            Pmusic[i] = 0.0  # Default set this point to 0 and wait for special handling later

    # Search spatial spectrum based on prior knowledge of source number to find DOA
    # Adjust DOA values for angle range of -90 to 90
    Doa = find_peaks(Pmusic, sig_num) * angle_gap - 90
    return Doa