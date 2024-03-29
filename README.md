# EEGNeX

<a href="https://github.com/chenxiachan/EEGNeX/blob/main/LICENSE.md"><img alt="GitHub license" src="https://img.shields.io/github/license/chenxiachan/EEGNeX"></a>
<a href="https://github.com/chenxiachan/EEGNeX/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/chenxiachan/EEGNeX?style=social"></a>
<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fchenxiachan%2FEEGNeX"><img alt="Twitter" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fchenxiachan"></a>

About Open source code of paper: <br>*-Toward reliable signals decoding for electroencephalogram: A benchmark study to EEGNeX*.<br>
-https://arxiv.org/abs/2207.12369


---
![1](https://github.com/chenxiachan/EEGNeX/assets/106488602/d332538a-12a7-42af-bdd8-12583e85652e)


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

![4](https://github.com/chenxiachan/EEGNeX/assets/106488602/f5ff3de7-bf65-4cc6-803b-8786216a4d29)
<br>
<i>EEGNeX architecture</i>

![5](https://github.com/chenxiachan/EEGNeX/assets/106488602/8ccbdd55-d9df-46b4-aa99-94ddac7b11fc)

---

More models are planned to be added:
- DeepConvNet
- ShallowConvNet
- SNN(Spike neural network)_based models

We also welcome you to contribute any model resources/papers in the discussion for our future plan :)
