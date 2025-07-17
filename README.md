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

After downloading, the light curves undergo a preprocessing pipeline to ensure consistency and quality for model training. This is implemented in the `normalize_curves` function (in `function.py`). The steps involved are as follows:

- **NaN Removal and Normalization**:  
  Each curve is first cleaned and normalized via:
  ```python
  lc = curve.remove_nans().normalize()
  ```
  This removes missing values (NaNs) and normalizes the flux by dividing it by the median, ensuring the curves are on a comparable scale.

- **Outlier Removal via Sigma-3 Clipping**:  
  A 3σ clipping technique is applied to eliminate extreme flux values:
  ```python
  mean, std = np.mean(flux), np.std(flux)
  flux = np.clip(flux, mean - 3*std, mean + 3*std)
  ```
  This retains flux values within three standard deviations of the mean, assuming an approximately normal distribution.

- **Denoising with Median Filter**:  
  A simple moving average filter smooths the signal to reduce noise:
  ```python
  flux = np.convolve(flux, np.ones(5)/5, mode='same')
  ```

- **Length Normalization (Padding/Truncation)**:  
  All light curves are adjusted to a fixed length to ensure uniform input size:
  ```python
  if len(flux) > padding_length:
      flux = flux[:padding_length]
  else:
      flux = np.pad(flux, (0, padding_length - len(flux)), mode='edge')
  ```
  Truncation is applied to longer sequences, while shorter ones are padded using edge values to avoid introducing artificial trends.

The processed flux arrays are then stored as NumPy arrays and collectively form the final dataset used for training and evaluation.

### Creation of the dataset



### Model



### Testing procedure



### Results
























