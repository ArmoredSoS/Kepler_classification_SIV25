# A minimal approach to Kepler Objects of Interest classification with 1D CNN

## Objective and approach:

  - The main objective of this project is to create a minimal model, using standard components, that performs well in classificating light curves from the Kepler dataset.  
    This approach leads to a refined, lightweight, easy-to-run and implement model, that although has not state-of-the-art performance, still gives satisfactory results
    that can be utilized as a first scrutiny when analyzing planet transit data.
  - The choice of language for the project is (obviously) Python, thanks to its plentiful libraries for machine learning applications.
  - The model choosen for the project is a 1D CNN, wo which then is added:
    - Dropout layers
    - Batch normalization
    - Max pooling
    - Global average pooling
    - Flatten
    - Fully connected layer (linear output)

## 
