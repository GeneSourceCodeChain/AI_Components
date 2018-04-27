# AI-components
### Introduction
AI Components of GeneSourceCode project is a subproject taking charge of machine learning related tasks. AI Components aim to make full use of gene and other medical data with the facilities of modern AI technologies. AI Components currently focus on 

1.Prediction of diseases and traits directly from raw DNA sequence.
We will test both traditional classification/regression algorithm and popular deep neural network ways such as LSTM, p-LSTM, IndRNN, attention model and so on to process raw DNA sequential data.

2.Prediction of diseases and traits from hand-designed feature.
The hand-designed feature extracted from raw DNA, RNA or histone sometime may be discriminative enough to make prediction task viable. We will try to extract and learn on features this way.

3.Medical application based on visual clues.
Computer Vision has become a reliable way of prediction after deep learning prevails. Medical scientists have adopted this method to various applications such as predicting or detecting certain diseases, image processing on X ray pictures, and so on. We will implement all these applications in this subproject and make them optional service modules.

4.Mining fitness status on physical examination and motion data,
We will also mining data provided by users to detect potential fitness problem or reveal healthy status. 

### Build everything
Build everything with command
```Bash
make
```
The project strongly relies on functions provided from [Caffe2](https://github.com/caffe2/caffe2) and [Caffe2 C++ helper](https://github.com/breadbread1984/caffe2_cpp_tutorial) . So you need to install these libraries to compile tools in this repo.

### Components
1.Prediction of diseases and traits directly from raw DNA sequence.

(1)rawDNA/LSTM: classification base on DNA subsequence:

You can learn a classifier with train_LSTM. The dataset generation tools will be released soon.

2.Prediction of disease and trais from hand-designed feature.

3.Medical appliation based on visual clues.

(1)visual/facial: classification based on facial images:

You can learn a classifier with train_facial_classifier. The dataset generation tools will be released soon.

4.Mining fitness status on physical examination and motion data.
