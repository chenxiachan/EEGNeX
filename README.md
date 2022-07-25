# EEGNeX
About Open source code of paper: *Toward reliable signals decoding for electroencephalogram: A benchmark study to EEGNeX*.<br>

<a href="https://github.com/chenxiachan/EEGNeX/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/chenxiachan/EEGNeX"></a>

---
![image](https://user-images.githubusercontent.com/106488602/176918224-4b24bf92-109e-48e8-b74b-8f44c5ba76b4.png)


This notebook is released for easy implementation of running all benchmarkmodels designed for electroencephalography(EEG) classification tasks, including:<br>
- *Single_LSTM*
- *Single_GRU*
- *OneD_CNN*
- *OneD_CNN_Dilated*
- *OneD_CNN_Causal*
- *OneD_CNN_CausalDilated*
- *TwoD_CNN*
- *TwoD_CNN_Dilated*
- *TwoD_CNN_Separable*
- *TwoD_CNN_Depthwise*
- *CNN_LSTM*
- *CNN_GRU*
- *Single_ConvLSTM2D*
- *EEGNet_4_2*
- *EEGNet_8_2*
- *EEGNeX_8_32*

For running the code, please run notebook `Run_model.ipynb`

Additional python packages required:
- keras 2.8.0
- tensorflow 2.8.0
- torch 1.10.2

The result folder contains validation results of running benchmarkmodels on four EEG datasets from paper.<br> 

![image](https://user-images.githubusercontent.com/106488602/176917267-b70cc98f-3b3c-4e19-a38e-2abdbe43d78f.png)<br>
<i>EEGNeX architecture</i>

---

More models are planned to be added:
- DeepConvNet
- ShallowConvNet
- SNN(Spike neural network)_based models

We also welcome you to contribute any model resources/papers in the discussion for our future plan :)
