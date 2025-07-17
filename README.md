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
downloaded_curves.extend(lc)
```
