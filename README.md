# ecg-ml

![intro img](https://github.com/jyu72/ecg-ml/blob/master/intro-img.png)

## Introduction

This is a Python library of tools for training and testing convolutional neural networks (CNNs)  to identify certain arrhythmias in 2-lead ECG rhythms. Tools are also provided for prediction on unannotated data and visualization of results.

## Requirements

- Python 2.7+, 3.5
- Keras
- Numpy
- Matplotlib
- [WFDB](https://pypi.org/project/wfdb/)

## Data

ECG signal, annotation, and header files can be downloaded from the [MIT-BIH Arrhythmia Database](https://www.physionet.org/physiobank/database/mitdb/). For any desired record, the signal (```.dat```), annotation (```.atr```), and header (```.hea```) files should all be downloaded and placed into the ```data``` folder.

## Installing

Clone this repository to your local machine using: https://github.com/jyu72/ecg-ml

```
$ git clone https://github.com/jyu72/ecg-ml
```

## Usage and Examples

The following is a guide to using the provided libraries to train and test CNN models and to make and visualize predictions using these models. We will perform our analysis on **premature ventricular contractions** (PVCs), a type of abnormal heart beat.

### Training
The following code segment trains a 2-dimensional CNN on the first 10800 samples (equivalent to 30 seconds) of ECG Record 119 (```119.dat```), using its Annotation file (```119.atr```), both of which can be downloaded from the [MIT-BIH Arrhythmia Database](https://www.physionet.org/physiobank/database/mitdb/) mentioned previously. We have opted to detect PVCs (```'V'```).

```python
import ecg_data
import ecg_cnn

model = ecg_cnn.train_2d([119], sampto=10800, beat_types=['V'], window_radius=24)
```

### Predicting

Trained CNN models can be used to make predictions on PVC locations in unseen, unannotated data.

```python
centers, windows = ecg_data.load_windows_2d(119, sampfrom=20000, sampto=23600, window_radius=24)
predictions = ecg_cnn.predict_2d(model, centers, windows, window_radius=24)
```

These predictions can be visualized as follows.

```python
annotation = ecg_data.create_annotation(centers, predictions, sampfrom=20000, sampto=23600, beat_types=['V'])
ecg_data.plot_prediction(119, annotation, sampfrom=20000, sampto=23600)
```

![prediction result](https://github.com/jyu72/ecg-ml/blob/master/demo-img.png)

A similar analysis can be performed on ECG Record 208, which can be visualized as follows.

![prediction result 2](https://github.com/jyu72/ecg-ml/blob/master/demo-img2.png)

We find that a few samples between the 63-second and 64-second marks are predicted incorrectly, but the overall performance is still acceptable.

As is shown below, *noisy* signal data does not preclude a model from making accurate predictions.

![prediction result 2](https://github.com/jyu72/ecg-ml/blob/master/demo-img3.png)

## Conclusion

As is demonstrated by the previous examples, it is possible to accurately identify **premature ventricular contractions** in individual patients using **convolutional neural network** models trained on previous data from the respective patient, even in the presence of significant signal noise. 

In the near future, I plan to present additional results using models trained on multiple patients to make predictions on data from previously unseen patients.

## Author

Jonathan Yu

## Acknowledgments

* Thank you to the MIT Laboratory for Computational Physiology (MIT-LCP) for their waveform-database ([WFDB](https://pypi.org/project/wfdb/)) Python package.
* This project was inspired by my time working as an EMT!
