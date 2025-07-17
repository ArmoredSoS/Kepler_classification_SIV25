# A Minimalist Approach to Kepler Objects of Interest Classification Using 1D Convolutional Neural Networks

## Objective and Methodology

The primary objective of this project is to develop a lightweight and efficient model for classifying light curves from the Kepler dataset. The aim is to maintain a minimal architecture that is straightforward to implement and computationally inexpensive, while still achieving reliable performance. Although the model is not intended to reach state-of-the-art accuracy, it serves effectively as a first-pass filter in the analysis of planetary transit data.

The project is implemented in Python, leveraging its extensive ecosystem of libraries for machine learning and scientific computing.

The core model is a one-dimensional convolutional neural network (1D CNN), enhanced with the following architectural components:

- **Dropout layers**: to mitigate overfitting by randomly deactivating neurons during training  
- **Batch normalization**: to stabilize and accelerate training by normalizing layer inputs  
- **Max pooling**: to reduce dimensionality while preserving salient features  
- **Global average pooling**: to condense feature maps before classification  
- **Flattening layer**: to transform the pooled features into a vector  
- **Fully connected (linear) layer**: to produce the final classification output

## Implementation and code review

### Retrieving and downloading the dataset

To retrieve the dataset, Kepler IDs are first obtained using the ```astroquery``` library, which provides direct access to the NASA Exoplanet Archive, including data from the Kepler mission. 
The confirmed Kepler Objects of Interest (KOIs) are extracted using the ```query_criteria function```. This function supports a simplified SQL-like syntax for querying archive tables: 
```
KeplerIDs = NasaExoplanetArchive.query_criteria( table = "cumulative", select = "kepid, koi_disposition", where = "koi_disposition = 'CONFIRMED'")
```

Once the Kepler IDs are retrieved, the ```lightkurve``` library is used to download the corresponding light curves. 
Specifically, the ```search_lightcurve``` function retrieves available light curve data for a given Kepler ID, while ```download_all downloads``` all matching data to a specified local directory:
```
lc = search_lightcurve(f"KIC {ID}", mission="Kepler").download_all(download_dir=download_dir)  
```

The downloaded curves are saved in the specified folder as folders representing the various targets, each containing roughly 10-15 curves.

### Preprocessing

The curves after downloading have to be normalized, padded or truncated and denoised. For this thre is the ```normalize_curves``` function in ```function.py```.  
The process applies:  
  - **Removal of NaNs and normalization**: ```lc = curve.remove_nans().normalize()``` removes NaNs and normalizes the function by dividing the flux by the median flux
  - **Outlier clipping through sigma-3 clipping**: computes mean and standard deviation, defines a threshold and then remove data points outside the threshold-defined range
    ```
    mean, std = np.mean(flux), np.std(flux)
    flux = np.clip(flux, mean - 3*std, mean + 3*std)
    ```
  - **Denoising**: done through a simple median filter
    ```
    flux = np.convolve(flux, np.ones(5)/5, mode='same')
    ```
  - **Padding/truncation**: done to homogenize the sizes of light curves
    ```
    if len(flux) > padding_length:
      flux = flux[:padding_length]
    else:
      #Using edge mode to limit the artificial creation of dips or spikes in the data when padding
      flux = np.pad(flux, (0, padding_length - len(flux)), mode='edge') 
    ```
























