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
- **Weight initiliazation**: using Kaiming normal distribution 

## Implementation and code review

### Retrieving and downloading the dataset

To retrieve the dataset, Kepler IDs are first obtained using the `astroquery` library, which provides direct access to the NASA Exoplanet Archive, including data from the Kepler mission. 
The confirmed Kepler Objects of Interest (KOIs) are extracted using the `query_criteria function`. This function supports a simplified SQL-like syntax for querying archive tables: 
```python
KeplerIDs = NasaExoplanetArchive.query_criteria( table = "cumulative", select = "kepid, koi_disposition", where = "koi_disposition = 'CONFIRMED'")
```

Once the Kepler IDs are retrieved, the `lightkurve` library is used to download the corresponding light curves. 
Specifically, the `search_lightcurve` function retrieves available light curve data for a given Kepler ID, while `download_all downloads` all matching data to a specified local directory:
```python
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

For the dataset construction, it is important to note that the classification task is framed as a binary problem. Specifically, Kepler Objects of Interest (KOIs) with a `CONFIRMED` disposition are treated as **positive examples**, while those labeled as `FALSE POSITIVE` are treated as **negative examples**. Accordingly, the labels are assigned as follows:
- `CONFIRMED` → `1` (positive class)  
- `FALSE POSITIVE` → `0` (negative class)


The normalized light curves are aggregated into a list of NumPy arrays, forming the dataset to be used in model training. PyTorch is then employed to construct `DataLoader` objects and to split the data into training and testing subsets via the `torch_dataset` function:
```python
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_ratio, stratify=labels)
train_set = KeplerDataset(data_train, labels_train)
test_set = KeplerDataset(data_test, labels_test)
```

The `KeplerDataset` class is a custom dataset wrapper tailored for Kepler light curve data. It adheres to the standard PyTorch `Dataset` interface but includes a reshaping operation to ensure compatibility with the 1D CNN input format. In particular, an additional channel dimension is introduced:
```python
self.data = torch.tensor(data[:, np.newaxis, :], dtype=torch.float32)
self.labels = torch.tensor(labels, dtype=torch.int64)
```

This reshaping step is critical to match the expected input shape of PyTorch convolutional layers `[batch_size, channels, sequence_length]`.

## Model

The model implemented is a 1D Convolutional Neural Network (CNN). The architecture is structured as follows:

- **Convolutional Layer**: A 1D convolution with configurable input channels, output channels, kernel size, and stride to extract features from the input time series.
- **Dropout Layers**: applied to reduce overfitting by randomly deactivating neurons during training.
- **Batch Normalization**: used after dropout to normalize activations, improving training stability and convergence.
- **Activation Function**: customizable non-linearity applied after normalization.
- **Pooling Layers**: max pooling reduces temporal resolution while preserving important features; adaptive average pooling reduces output dimension to a fixed size.
- **Flatten Layer**: converts the pooled feature maps into a 1D vector.
- **Fully Connected Layer**: outputs class scores for binary classification.

An optional weight initialization function can be applied during model instantiation to improve training dynamics. The model architecture is kept minimal to avoid overfitting, especially when training on limited datasets.
```python
self.conv = nn.Sequential(    
  nn.Conv1d(input_ch, out_ch, kernel_size, stride), #Convolutional layer
  nn.Dropout(p = dropout_rate), #Dropout layer after convolutional
  nn.BatchNorm1d(out_ch), #Normalization
  activation(), #Tunable activation function
 
  nn.MaxPool1d(pool_kernel),
  nn.AdaptiveAvgPool1d(1),
  nn.Flatten(),

  nn.Dropout(p = dropout_rate), #Dropout layer before linear
  nn.Linear(out_ch, output_dim)                
)      
if init_weight: init_weights(self)
```

## Training and testing procedure

The model is trained and evaluated over 100 epochs. The optimizer used is `AdamW`, chosen for its robustness in handling weight decay and stability during training.  
A learning rate scheduler (`ReduceLROnPlateau`) is used to lower the learning rate when the validation loss stagnates, helping the model converge.

During each epoch, the following five evaluation metrics are computed:
- **Accuracy**: Measures the overall proportion of correct predictions.
- **Precision**: Indicates the model's ability to avoid false positives.
- **Recall**: Indicates the model's ability to detect true positives (avoiding false negatives).
- **F1-score**: Harmonic mean of precision and recall, offering a balanced measure.
- **ROC AUC**: Evaluates the model's ability to distinguish between classes (separability).

After training, the epoch that achieved the best (lowest) validation loss is identified, and the corresponding metrics are reported as the model's best performance.

### Training loop
noise
### Evaluation loop

## Results
























