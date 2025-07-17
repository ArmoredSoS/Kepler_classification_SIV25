# A Minimalist Approach to Kepler Objects of Interest Classification Using a 1D Convolutional Neural Network

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
  
- **Determining Optimal Length for Padding/Truncation**:
  To select a length that minimizes information loss, the distribution of light curve lengths is analyzed:
  ```python
  curves = download_curves(get_kepids('all'), 10)
  normalized = normalize_curves(curves)
  lengths = [len(curve.flux) for curve in normalized]
  
  plt.hist(lengths, bins=50)
  plt.xlabel("Time series length")
  plt.ylabel("Frequency")
  plt.title("Distribution of Kepler light curve lengths")
  plt.show()
  ```
  The analysis indicated that a length of approximately 65,000 samples offers a reasonable trade-off.
  
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

## Training and testing procedures

The model is trained and evaluated over 100 epochs. The optimizer used is `AdamW`, chosen for its robustness in handling weight decay and stability during training.  
A learning rate scheduler (`ReduceLROnPlateau`) is used to lower the learning rate when the validation loss stagnates, helping the model converge.

During each epoch, the following five evaluation metrics are computed:
- **Accuracy**: Measures the overall proportion of correct predictions.
- **Precision**: Indicates the model's ability to avoid false positives.
- **Recall**: Indicates the model's ability to detect true positives (avoiding false negatives).
- **F1-score**: Harmonic mean of precision and recall, offering a balanced measure.

Early stopping is used in order to stop the model if there are no improvements, saving inference time:
```python
if loss < best_loss:
  best_loss = loss
  best_metrics = evaluation
  best_epoch = epoch + 1
else:
  stale += 1
  if stale >= patience:
    break #Not elegant but it works
```

After training, the epoch that achieved the best (lowest) validation loss is identified, and the corresponding metrics are reported as the model's best performance.

### Training loop

The training loop follows a standard supervised learning pipeline. The `CrossEntropyLoss` function is used, which is well-suited for classification tasks as it penalizes incorrect class predictions with respect to the true label probabilities.

Furthermore, also an augmentation step is added during training, injecting additional noise into the data using `randn_like()` and controlled by a noise level:
```python
noise = torch.randn_like(batch) * noise_lvl
noisy_data = batch + noise
return noisy_data
```

At each epoch, the model is trained on batches of the training dataset, the size of which is kept at 128 (because of hardware limitations). After completing an epoch, the average **training loss** is computed and printed. This provides a basic diagnostic to monitor convergence and detect potential issues such as underfitting or overfitting.  

### Evaluation loop

The evaluation function follows too a standard pipeline. The model is switched to evaluation mode and inference is run without gradient computation. It iterates through the test data loader, computes the outputs, derives predictions, and accumulates the loss and labels. The average loss is computed over all test samples.

At the end, the function returns both the average loss and the pre-defined set of evaluation metrics, which are computed from the full set of predicted and true labels.

## Results

## Experimental Results

### Overview

We conducted a series of 22 tests on the `KeplerCNN` model, progressively evaluating the effects of various hyperparameters and architectural choices. These include activation functions, dropout and normalization layers, dataset size, learning rate, learning rate scheduler configuration, and noise augmentation.

---

### Activation Function Comparison

| Test | Activation | Accuracy | Precision | Recall | F1 Score |
|------|------------|----------|-----------|--------|----------|
| 1    | LeakyReLU  | 0.557    | 0.754     | 0.394  | 0.517    |
| 2    | ReLU       | 0.603    | 0.603     | 1.000  | 0.752    |
| 3    | ELU        | 0.603    | 0.603     | 1.000  | 0.752    |

**Observation:**  
LeakyReLU yields more balanced precision and recall, while ReLU and ELU give perfect recall but poor separability and overly optimistic F1 scores.

---

### Dropout Layer Placement

| Test | Dropout Placement       | Accuracy | Precision | Recall | F1 Score |
|------|--------------------------|----------|-----------|--------|----------|
| 4    | After linear only        | 0.594    | 0.605     | 0.939  | 0.736    |
| 5    | No dropout               | 0.598    | 0.608     | 0.939  | 0.738    |

**Observation:**  
Removing dropout layers slightly improves metrics, but maintaining some dropout seems beneficial for regularization.

---

### Normalization Layers

| Test | Normalization Config    | Accuracy | Precision | Recall | F1 Score |
|------|--------------------------|----------|-----------|--------|----------|
| 6    | None                     | 0.603    | 0.603     | 1.000  | 0.752    |
| 7    | Additional norm + linear | 0.603    | 0.603     | 1.000  | 0.752    |
| 8    | Same as 7, larger data   | 0.600    | 0.600     | 1.000  | 0.750    |

**Observation:**  
Normalization has minimal measurable effect in this configuration.

---

### Dropout Rate Tuning (with normalization and larger dataset)

| Test | Dropout Rate | Accuracy | Precision | Recall | F1 Score |
|------|--------------|----------|-----------|--------|----------|
| 9    | 0.3          | 0.600    | 0.600     | 1.000  | 0.750    |
| 10   | 0.5          | 0.582    | 0.697     | 0.538  | 0.607    |
| 11   | 0.7          | 0.600    | 0.600     | 1.000  | 0.750    |
| 12   | 0.6          | 0.420    | 0.514     | 0.631  | 0.566    |

**Observation:**  
Dropout = 0.5 offers the best trade-off; too much or too little harms recall or precision.

---

### Learning Rate and Scheduler Tuning

| Test | Dataset Size | LR     | Accuracy | Precision | Recall | F1 Score |
|------|--------------|--------|----------|-----------|--------|----------|
| 13   | 100          | 1e-3   | 0.600    | 0.600     | 1.000  | 0.750    |
| 14   | 100          | 5e-3   | 0.600    | 0.600     | 1.000  | 0.750    |
| 15   | 100          | 3e-4   | 0.600    | 0.600     | 1.000  | 0.750    |
| 16   | 100          | 1e-5   | 0.600    | 0.600     | 1.000  | 0.750    |

**Observation:**  
Model is insensitive to LR between 1e-5 and 5e-3 on the 100-sample dataset.

---

### Effect of Dataset Size

| Test | Dataset | Accuracy | Precision | Recall | F1 Score |
|------|---------|----------|-----------|--------|----------|
| 17   | 200     | 0.402    | 0.511     | 0.596  | 0.550    |
| 18   | 200     | 0.613    | 0.613     | 1.000  | 0.760    |
| 19   | 200     | 0.611    | 0.875     | 0.425  | 0.572    |
| 20   | 200     | 0.613    | 0.613     | 1.000  | 0.760    |
| 21   | 200     | 0.577    | 0.851     | 0.375  | 0.521    |

**Observation:**  
Using a larger dataset improves generalization but also increases variability. Low LR (1e-5) can underfit, but going too high (5e-3) gives inconsistent recall.

---

### Effect of Noise Augmentation

| Test | Noise Augmentation | Accuracy | Precision | Recall | F1 Score |
|------|---------------------|----------|-----------|--------|----------|
| 22   | no augmentation | 0.613    | 0.613     | 1.000  | 0.760    |
| 18   | augmentation   | 0.610    | 0.875     | 0.425  | 0.572    |

**Observation:**  
Overall training behavior and suggests that noise augmentation improves model robustness and consistency.

---

### Summary

- **Best trade-off** is reached using LeakyReLU, dropout = 0.5, LR = 1e-3, with at least 200 samples.
- **Recall tends to dominate** when overfitting is controlled, indicating a strong bias towards the positive class.
- Further improvement likely requires a bigger dataset and the consequential tuning for the best results.

## Final thoughts and conclusion

Overall, considering the minimalistic architecture of the model and the small dataset (~3,400 light curves), the obtained results are satisfactory. 
Nonetheless, there is clear potential for improvement, particularly through the use of a larger dataset, which could better support deeper architectures. 
Preliminary tests with such an architecture incorporating extra convolutional and normalization layers—see `OverComplicatedCNN_Test` in `model.py`—resulted in poorer performance, 
likely due to overfitting or insufficient data to fully exploit the increased model capacity.

## Sources
  - https://astroquery.readthedocs.io/en/latest/ipac/nexsci/nasa_exoplanet_archive.html
  - https://astroquery.readthedocs.io/en/latest/api/astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchiveClass.html#rb480749af1eb-1
  - https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
  - https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
  - https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
  - https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
  - https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
  - https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
  - https://docs.pytorch.org (used for a bit of everything)
  - https://docs.astropy.org/en/stable/index.html
  - https://astroquery.readthedocs.io/en/latest/
  - https://scikit-learn.org/dev/user_guide.html    






















