### ***This is an On-going project***



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



[concept_detection_notebook](https://github.com/AdeboyeML/Imageclef_Concept_Detection/blob/master/concept_detection_full_roco_dataset.ipynb)


### -- Summary (models still needs further training...more compute power required)

- Evaluation metric == F1 Score: is the most suited for imbalanced class labels (in our case -- concepts to be detected).


##### - Decision Threshold was tuned on validation dataset, the best threshold was 0.1

No | Model Description | Dev. f1 Score | Test f1 Score
--- | --- | --- | ---
0 | DenseNet-121 Encoder + FFNN (AUEB NLP Group, 2019) | 0.157 | 0.146
1 | DenseNet-121 Encoder + k-NN Image Retrieval (AUEB NLP Group, 2019) | 0.147 | 0.142

### Scientific papers References

[AUEB NLP Group, 2019](http://ceur-ws.org/Vol-2380/paper_136.pdf)

[Pelka et al., 2019](http://ceur-ws.org/Vol-2380/paper_245.pdf)
