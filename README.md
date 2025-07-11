# CODECRAFT_ML_03
Pet-Image Binary Classification for Dogs and Cats.

Classification is a form of supervised machine learning in which the label represents a categorization, or class. There are two common classification scenarios-binary, multiclass
In binary classification, the label determines whether the observed item is (or isn't) an instance of a specific class. Or put another way, binary classification models predict one of two mutually exclusive outcomes.

This Repository contains Two methods for Image classification of dogs and cats:

 1)MACHINE LEARNING METHOD : by creating SUPPORT VECTOR MACHINE (SVM) Model

 2)DEEP LEARNNG METHOD : by creating CONVOLUTIONAL NEURAL NETWORK (CNN) Model
 
 Dataset Links:

1)for SVM: https://www.kaggle.com/datasets/salader/dogs-vs-cats

2)for CNN: https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD
 
| Feature                | **CNN (Convolutional Neural Network)**                                         | **SVM (Support Vector Machine)**                                                     |
| ---------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **Type**               | Deep Learning                                                                  | Machine Learning                                                                     |
| **Best for**           | Image classification, object detection, facial recognition, NLP (complex data) | Binary or multi-class classification of structured/tabular or small-scale image data |
| **Data Handling**      | Automatically extracts features from raw images (end-to-end learning)          | Requires manual feature extraction (pixels, textures, shapes) before classification  |
| **Scalability**        | Scales well with large datasets                                                | Struggles with very large datasets, especially high-dimensional ones                 |
| **Computation**        | High computational cost (GPU often required)                                   | Relatively less computationally intensive                                            |
| **Accuracy on Images** | Typically much higher due to deep feature extraction                           | Lower unless features are carefully engineered                                       |
| **Overfitting Risk**   | Lower when trained properly with regularization                                | High if too many features or small data                                              |
| **Speed**              | Slower training but faster inference once trained                              | Faster training on small datasets                                                    |
| **Interpretability**   | Harder to interpret (black box)                                                | Easier to interpret (decision boundaries)                                            |
