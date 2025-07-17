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

To retrieve the dataset, the kepler IDs were needed first, for this the ```astroquery``` library is used, providing a direct access to NASA exoplanet archive (and consequently Kepler data).  
The following is the code snippet used to retriev confirmed KOIs, thorugh the ```query_criteria``` function, providing a SQL-like syntax to access the table of object of interest.  
```
KeplerIDs = NasaExoplanetArchive.query_criteria( table = "cumulative", select = "kepid, koi_disposition", where = "koi_disposition = 'CONFIRMED'")
```

To download the curves, the ```lightkurve``` library is used, it provides the ```search_lightcurve``` function, that allows to find the light curves starting from the Kepler IDs retrieved.
Then the ```download_all``` function takes the search results and downloads the corresponding light curves to the local machine for processing.
```
lc = search_lightcurve(f"KIC {ID}", mission="Kepler").download_all(download_dir=download_dir)  
downloaded_curves.extend(lc)
```
