# NLP-Based-Digital-Pathology

This is an implementation of our paper ***Time to Embrace Natural Language Processing (NLP)-based Digital Pathology: Benchmarking NLP- and Convolutional Neural Network-based Deep Learning Pipelines***

### Pipeline of Our Methods

The pipeline of our method is shown in Pipeline Figure:

![Pipeline Figure](/figures/pipeline.png)

Pipelines for predicting of MSI, BRAF mutation, and CIMP in CRC. MCO-CRC and TCGA-CRC-DX were used to train and test for prediction of molecular biomarkers in CRC (i.e., MSI, BRAF mutation, and CIMP). The whole-slide images were tessellated into non-overlapping tiles of 512 × 512 pixels at a resolution of 0.5 µm. The resulting tiles were then resized to 224 × 224 pixels and color normalized. Tumor tissues (tiles) were subsequently selected by a Swin-T-based tissue-type classifier. Up to 500 tumor tiles were randomly selected for each slide. Five NLP-based models (in orange) and four CNN-based models (in blue) were trained to predict tile-level biomarkers. Models with a red star represent models that are applied in digital pathology the first time. The predictive slide labels were obtained via tile score aggregation.

### Requirements

We used two environments to implement our paper due to the different versions of 
** models without MobileViT

-``python 3.8``
-
