from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from lightkurve import search_lightcurve
from lightkurve import LightCurve
from typing import Literal
from astropy.table import Table
from typing import Literal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch


#References: https://astroquery.readthedocs.io/en/latest/ipac/nexsci/nasa_exoplanet_archive.html
#            https://astroquery.readthedocs.io/en/latest/api/astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchiveClass.html#rb480749af1eb-1
#            https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
#            https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
#            https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
#            https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

def download_curves(kepid: list[int], n_targets: int, download_dir: str = "light_curves") -> list[LightCurve]:
    """
    Downloads the light curves corresponding to the input Kepler IDs
    
    Args:
        kepid (list[int]): list of Kepler IDs
        n_targets (int): number of targets to download from the list, if equal or below 0, downloads all the targets
        
    Returns:
        download_curves (list[KeplerLightCurve]): list of downloaded kepler light curves
    """
    
    downloaded_curves: list[LightCurve] = [] 
    
    if n_targets > 0: kepid = kepid[:n_targets]
    
    print("Starting download")  
    for ID in kepid[:n_targets]:
        try:
            lc = search_lightcurve(f"KIC {ID}", mission="Kepler").download_all(download_dir=download_dir)
            downloaded_curves.extend(lc)
        except:
            print("Error downloading")
    print("Finished download")

    return downloaded_curves

def normalize_curves(curves: list[LightCurve], padding_length: int = 60000) -> np.ndarray:
    """
    Prepares kepler curves for processing (removal of NaNs, padding, splicing, normalization, denoising, outlier clipping), 
    outputs them as NumPy arrays for easier later processing 
    
    Args:
        curves (list[LightCurve]): Array of Kepler light curves to normalize
        padding_length (int (default = 60000)): Length at which curves will be either truncated or padded
    
    Returns:
        normalized_curves (np.ndarray): NumPy array of shape (n_curves, padding_length) ready for processing in ML pipeline 
    """

    normalized_curves = []
    
    for curve in curves:
        
        #1. Remove NaNs, normalize, flatten
        lc = curve.remove_nans().normalize()#.flatten()

        #2. Outlier clipping with sigma clipping (selected for ease of use)
        flux = lc.flux.value
        mean, std = np.mean(flux), np.std(flux)
        flux = np.clip(flux, mean - 3*std, mean + 3*std)

        #3. Denoising through a simple median filter
        flux = np.convolve(flux, np.ones(5)/5, mode='same')

        #4. Padding/Truncate
        if len(flux) > padding_length:
            flux = flux[:padding_length]
        else:
            #Using edge mode to limit the artificial creation of dips or spikes in the data when padding
            flux = np.pad(flux, (0, padding_length - len(flux)), mode='edge') 

        normalized_curves.append(flux)

    return np.array(normalized_curves)

def get_kepids(type: Literal["confirmed", "false", "all"]) -> list[int]: 
    """
    Returns the list of selected kepler IDs
    
    Args:
        type (Literal["confirmed", "false", "all"]):
        confirmed: KOIs with CONFIRMED disposition
        false: KOIs with FALSE POSITIVE disposition
        all: All KOIs
    
    Returns:
        KeplerIDs (list[int]): list of Kepler IDs
    """
    
    KeplerIDs: Table = Table()
    
    print("Retrieving IDs from table")
    if type == "confirmed":
        KeplerIDs = NasaExoplanetArchive.query_criteria( table = "cumulative", select = "kepid, koi_disposition", where = "koi_disposition = 'CONFIRMED'") #type: ignore
        
    if type == "false":
        KeplerIDs = NasaExoplanetArchive.query_criteria( table = "cumulative", select = "kepid, koi_disposition", where = "koi_disposition = 'FALSE POSITIVE'") #type: ignore
        
    if type == "all":
        KeplerIDs = NasaExoplanetArchive.query_criteria( table = "cumulative", select = "kepid") #type: ignore
    print("Finished retrieving IDs from table")

    return KeplerIDs["kepid"].data.tolist()

def training_loop(model: nn.Module, loader: data.DataLoader , optimizer: optim.Optimizer, 
                  criteria: nn.modules.loss._Loss, device: Literal["cpu","cuda"], noise = False):
    """
    Function for model training
    
    Args:
        model (torch.nn.Module): Model to use
        loader (torch.utils.data.DataLoader): Data loader to use
        criteria (nn.module.loss._Loss): Loss function to use
        device (Literal["cpu","cuda"]): Device to use
        noise (bool (default = False)): whether to apply additional noise to the data or not
    """
    
    model.to(device)
    model.train()
    
    running_loss : float = 0
    
    for data, labels in loader:
        
        if noise: data = noise_data(data)
        
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criteria(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"training loss: {running_loss}")

def evaluate_model(model: nn.Module, loader: data.DataLoader, criteria: nn.modules.loss._Loss, device: Literal["cpu","cuda"]):
    """
    Function for model evaluation
    
    Args:
        model (torch.nn.Module): Model to use
        loader (torch.utils.data.DataLoader): Data loader to use
        criteria (nn.module.loss._Loss): Loss function to use
        device (Literal["cpu","cuda"]): Device to use
        
    Returns:
        (metrics, avg_loss) (dict[str, Unknown], float): (Accuracy, precision, recall, f1_score, roc_auc), average loss
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    running_loss : float = 0
    total: float = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            #AI-generated snippet to track loss
            loss = criteria(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_preds),
    }

    return metrics, avg_loss
    
def noise_data(batch: Tensor, noise_lvl : float = 0.01) -> Tensor:
    """
    Function adding noise to the input tensor
    Args:
        batch (Tensor): input batch
        noise_lvl (float): modifier for the amount of noise to add with torch.randn_like
        
    Returns:
        noisy_data (Tensor): resulting noisy data
    """
    noise = torch.randn_like(batch) * noise_lvl
    noisy_data = batch + noise
    return noisy_data

def init_weights(m: nn.Module) -> None:
    """
    Function to initialize a layer's weights with a Kaiming normal distribution
    
    Args:
        m (nn.Module): model to initialize
    """
    
    #Using kaiming because it pairs well with activation functions such as ReLU, GELU or LeakyReLU
    #See: He, K. et al. (2015)
    
    #Using isinstance to apply initialization only on the right layers
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
