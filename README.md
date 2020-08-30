

# Imageclef_Concept_Detection
The aim of this task is to automatically detect medical concepts related to each image, as a first step towards generating image captions, medical reports, or to help in medical diagnosis.

### Steps Taken:

- Acquisition of Datasets and Extraction of Images from Tarfiles
- Data Exploration
- Data Analysis
- Data Visualization 
- Data Preprocessing
- Implementation of Machine learning models 
- Evaluation and Prediction --




### -- Summary (models still needs further training...more compute power required)

[concept_detection_notebook](https://github.com/AdeboyeML/Imageclef_Concept_Detection/blob/master/concept_detection_full_roco_dataset.ipynb)

### -- Full ROCO Dataset
No | Datasets | No of images
--- | --- | ---
0 | Train Dataset | 60963
1 | Validation Dataset | 7,700
2 | Test Dataset | 7,662
3 | Total | 76328


- Evaluation metric == F1 Score: is the most suited for imbalanced class labels (in our case -- concepts to be detected).


##### - Decision Threshold was tuned on validation dataset, the best threshold was 0.1

No | Model Description | Dev. f1 Score | Test f1 Score
--- | --- | --- | ---
0 | DenseNet-121 Encoder + FFNN (AUEB NLP Group, 2019) | 0.157 | 0.146
1 | DenseNet-121 Encoder + k-NN Image Retrieval (AUEB NLP Group, 2019) | 0.147 | 0.142




### -- Reduced Dataset
No | Datasets | No of images
--- | --- | ---
0 | Train Dataset | 30000
1 | Validation Dataset | 3500
2 | Test Dataset | 3500
3 | Total | 37000


### -- Summary ( All models still needs retraining)

##### - Decision Threshold was tuned on validation dataset, the best threshold was 0.1

No | Model Description | Dev. f1 Score | Test f1 Score
--- | --- | --- | ---
0 | DenseNet-121 Encoder + FFNN (AUEB NLP Group, 2019) | 0.168 | 0.161
1 | Resnet 101 + FFNN, Multi-label classification in Xu, et al 2019 | 0.168 | 0.160
2 | DenseNet-121 Encoder + k-NN Image Retrieval (AUEB NLP Group, 2019) | 0.150 | 0.142
4 | ResNet 101 + Data Filtering (Df1) -- Xu et al., 2019 (Damo Group) | 0.169 | 0.160
5 | ResNet 101 + Data Filtering (Df3) -- Xu et al., 2019 (Damo Group) | 0.170 | 0.163



### Python Scripts

- DenseNet-121 Encoder/Resnet 101 + FFNN - train_model_get_threshold.py,
- DenseNet-121 Encoder + k-NN Image Retrieval - knn_train_test.py,
- ResNet 101 + Data Filtering (Df1/Df3) - filtered_model.py,
- make predictions on test data - make_predictions.py,



### Scientific papers References

[AUEB NLP Group, 2019](http://ceur-ws.org/Vol-2380/paper_136.pdf)

[Pelka et al., 2019](http://ceur-ws.org/Vol-2380/paper_245.pdf)
