<p align="center"><img align="center" width="280" src="docs/img/icon.png"/></p>
<h1 align="center">ChronomancerAI 

A TimeSeries analysis and model training repository</h1>
<hr>

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,pytorch" />
  </a>
</p>
<hr>

ChronomancerAI is a repository dedicated to time series data analysis and model training. It contains two main projects: ECG anomaly detection and wind energy production prediction. Each project is implemented using LSTMs and other deep learning models to analyze and predict patterns within the respective datasets.

# Description of Notebooks

## Wind Turbine Anomaly Detection model

This notebook is designed to find anomaly in Wind Turbine Sensors using LSTMs. The dataset used is ["CARE to Compare"](https://arxiv.org/abs/2404.10320) and contain multiple sensors over multiple months for 3 Wind Farms.

![Anomaly Detection Graph](docs/img/care_to_compare_graph.png)

### Key Steps:

- Data preprocessing: Load the sensor measurements from all the Wind Farms (oil temperature, gearbox temperature, wind spee,d power measurement...).
- Model building: Create an LSTM-based model to predict anomaly detections.
- Training: Train the model on historical sensor data.
- Evaluation: Compare the model's predictions with the actual energy values.

### Results

![Wind Farm Anomaly Detection Example](docs/img/windfarm_anomaly_detection.png)

- The plot shows anomaly scores on a dataset without anomaly (blue) and on a dataset with anomalies (orange). The model demonstrates good capability in anomaly detection, and allows multiple warning levels.

## Anomaly Detection for ECG using LSTM

This notebook is used to train and evaluate an LSTM-based autoencoder model to detect anomalies in ECG signals. The dataset used is ECG5000, which contains multiple examples of normal and abnormal heartbeats. The model is trained to reconstruct normal signals, and the reconstruction loss is used as an indicator of anomalies.

### Key Steps:

- Data preprocessing: Load and normalize the ECG data.
- Model building: Create an LSTM-based autoencoder model.
- Training: Train the model using normal ECG samples.
- Evaluation: Compute reconstruction losses and use them to identify anomalies.

### Results

![Normal and Anomaly Classification Examples](docs/img/anomaly_detection_examples.png)

- The plots display the reconstructed ECG signals for both normal and anomalous cases. Each plot has two lines: `true` (the original signal) and `reconstructed` (the reconstructed signal from the model).
- The `loss` value in each title represents the reconstruction error. A higher loss usually indicates an anomaly.

## Wind Power Prediction using LSTM AutoEncoder

This notebook is designed to predict wind energy production using LSTMs. The dataset used (`T1.csv`) contains wind energy production values over time, and the LSTM model learns to predict future energy output based on historical data.

### Key Steps:

- Data preprocessing: Load the wind energy dataset and normalize the values.
- Model building: Create an LSTM-based model to predict future energy production.
- Training: Train the model on historical energy data.
- Evaluation: Compare the model's predictions with the actual energy values.

### Results

![Wind Power Prediction Example](docs/img/prediction_examples.png)

- The plot shows the ground truth (`True`) versus the predicted (`Predicted`) wind energy production over a period of time. The model demonstrates good predictive capability, capturing the general trend of the true data.

# Installation

To install the necessary dependencies, run the following command:

```sh
pip install -r requirements.txt
```

# Usage

Each notebook can be run independently for its respective task. Make sure to extract the datasets and place them in the appropriate directories.

# License

This repository is licensed under the MIT License. See the [Licence](LICENSE) file for more details.


