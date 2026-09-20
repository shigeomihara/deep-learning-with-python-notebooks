import numpy as np

def naive(dataFile):
    print("Naive Method:")
    total_abs_err = 0.0
    samples_seen = 0
    for samples, targets in dataFile.val_dataset:
        preds = samples[:, -1, 1] * dataFile.std[1] + dataFile.mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    print(f"  MAE for val = {total_abs_err / samples_seen: .2f} (degC)")
    
    total_abs_err = 0.0
    samples_seen = 0
    for samples, targets in dataFile.test_dataset:
        preds = samples[:, -1, 1] * dataFile.std[1] + dataFile.mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    print(f"  MAE for test = {total_abs_err / samples_seen: .2f} (degC)")
    
    
