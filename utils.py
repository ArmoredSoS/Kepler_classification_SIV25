import numpy as np
import torch

from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
from function import *

def create_dataset(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of Kepler light curves
    
    Args:
        n_samples (int): Number of samples for the dataset
        
    Returns:
        (data, labels) (tuple[np.ndarray, np.ndarray]): Data-labels pair
    """
    #1. Get IDs of confirmed and not confirmed exoplanets
    confirmed_KID = get_kepids("confirmed")
    false_KID = get_kepids("false")

    #2. Download the corresponding light curves
    confirmed_curves = download_curves(confirmed_KID, n_samples // 2)
    notconfirmed_curves = download_curves(false_KID, n_samples // 2)

    #3. Normalized the obtained curves and transform in numpy arrays
    confirmed_curves_normalized = normalize_curves(confirmed_curves)
    false_curves_normalized = normalize_curves(notconfirmed_curves)

    #4. Put together the curves to create the dataset
    data = np.concatenate((confirmed_curves_normalized, false_curves_normalized), axis=0)
    labels = np.concatenate((np.ones(len(confirmed_curves_normalized)), np.zeros(len(false_curves_normalized))))
    return data, labels

class KeplerDataset(Dataset):
    """
    Class for the torch dataset with kepler data
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data[:, np.newaxis, :], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] #unsqueeze

def torch_datasets(KeplerData: tuple[np.ndarray, np.ndarray], test_ratio: float = 0.2):
    """
    Function for the creation of the torch dataset
    
    Args:
        KeplerData (tuple[np.ndarray, np.ndarray]): Input data as NumPy array
        test_ratio (float (default = 0.2)): Test/training data ratio for splitting of the dataset
                
    Returns:
        (train_set, test_set) (list[Subset[]]): Training and testing subsets     
    """
    data, labels = KeplerData
    
    # dataset = KeplerDataset(data, labels)
    # n_total = len(dataset)
    # n_test = math.ceil(n_total * test_ratio)
    # n_train = n_total - n_test
    # train_set, test_set = random_split(dataset, [n_train, n_test])
    
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_ratio, stratify=labels)
    
    train_set = KeplerDataset(data_train, labels_train)
    test_set = KeplerDataset(data_test, labels_test)
    
    return train_set, test_set

