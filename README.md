# Analog_SNN_RFLocalization

# RF Source Localization and Neuromorphic System

This repository contains code and resources for RF source localization dataset creation and analog-based spiking neural network synthesis with deep learning. These resources are associated with the paper "An End-to-End Analog Neuromorphic System for Low-Power RF Localization in IoT."

## Repository Structure

### 1. Simulated_Measured_Datasets_Generation

- **RF Environment Setup**:
  - The RF environment setup considered for dataset generation is based on the RF setup shown in Fig.
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
