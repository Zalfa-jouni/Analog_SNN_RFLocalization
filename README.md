# Analog_SNN_RFLocalization

This repository contains code and resources for RF source localization dataset creation and analog-based spiking neural network synthesis with deep learning. These resources are associated with the paper "An End-to-End Analog Neuromorphic System for Low-Power RF Localization in IoT."

## Repository Structure

### 1. Simulated_Measured_Datasets_Generation

- **RF Environment Setup**:
  - The RF environment setup considered for dataset generation is based on the RF setup shown in the figure below.
<img src="https://github.com/Zalfa-jouni/Analog_SNN_RFLocalization/blob/main/Images/RF_config.png" alt="RF Setup" width="500"/>
In this setup, the mobile source, marked by red crosses, can occupy any position along three concentric circles representing distances of 0.1, 0.23, and 0.5 meters from the origin. These circles allow for a full 360-degree positional coverage of the source, with the angle relative to the origin (θs) varying from 0 to 360 degrees in 10-degree increments. There are four receivers, each located at key points marked by blue squares to ensure full coverage and enhance localization accuracy within the 2D layout. The receivers are positioned at the midpoint of each boundary of the plane: Receiver 1 to the right, Receiver 2 at the top, Receiver 3 on the left, and Receiver 4 at the bottom, all maintaining an equal distance of 1 meter (dr) from the origin.
    
- **Simulated Dataset Generation Code**:
  - This section includes the code for generating the simulated dataset for RF localization.
  - You can modify various RF parameters such as:
    - Operating frequency
    - Antenna gain
    - Transmit power
    - Angular resolution
    - And many other parameters

- **Pseudocode for Simulated Dataset Generation**:
  - A pseudocode guide is provided to help understand the process of simulated dataset generation.

- **Measured Datasets**:
  - This section contains three files representing datasets generated from anechoic chamber measurements at different SNR levels:
    - SNR = 0 dB
    - SNR = 10 dB
    - SNR = 20 dB
  The measurement setup taken in consideration to develop the measured dataset is shown in the figure below.
<img src="https://github.com/Zalfa-jouni/Analog_SNN_RFLocalization/blob/main/Images/Experimental_Setup.png" alt="RF Setup" width="500"/>


### 2. Analog_SNN_Learning_Synthesis

- **Analog SNN Learning and Synthesis Code**:
  - This section includes the code for synthesizing the analog SNN for deep learning using the provided datasets and the analog neurons post-layout properties.
  - A pseudocode guide is also provided to clarify the steps of the code.

## Installation

To use the code in this repository, you need the following requirements:

- MATLAB R2022a for simulated dataset generation.
- Python 3.10 with the following packages:
  ```python
  import tensorflow as tf
  import numpy as np
  import random
  import os
  import pandas as pd
  from scipy.interpolate import interp1d
  import scipy.io
  from sklearn.preprocessing import StandardScaler, MinMaxScaler
  from sklearn.model_selection import train_test_split
  import keras
  from keras.models import Model
  from keras.layers import Dense, Input, Dropout, Concatenate
  from keras.optimizers import Adam, SGD
  import matplotlib
  import matplotlib.pyplot as plt

## Note

The MATLAB code can be easily adjusted to use in Python as well.

## Usage

### Simulated Dataset Generation

To generate the simulated dataset, run the following command in MATLAB:
```
run('Generate_Simulated_Dataset.m')
```

### Analog SNN Learning and Synthesis

To synthesize the analog SNN using the provided datasets and the analog neurons properties, run the following command in Python:
```
python Analog_SNN_Learning.py
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Contact

Feel free to reach out if you have any questions or need further assistance. You can contact us at zalfa@ieee.org.
