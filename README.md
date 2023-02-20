# NLP-Based-Digital-Pathology

This is an implementation of our paper ***Time to Embrace Natural Language Processing (NLP)-based Digital Pathology: Benchmarking NLP- and Convolutional Neural Network-based Deep Learning Pipelines***

## Pipeline of Our Methods

The pipeline of our method is shown in Pipeline Figure:

![Pipeline Figure](/figures/pipeline.png)

Pipelines for predicting of MSI, BRAF mutation, and CIMP in CRC. MCO-CRC and TCGA-CRC-DX were used to train and test for prediction of molecular biomarkers in CRC (i.e., MSI, BRAF mutation, and CIMP). The whole-slide images were tessellated into non-overlapping tiles of 512 × 512 pixels at a resolution of 0.5 µm. The resulting tiles were then resized to 224 × 224 pixels and color normalized. Tumor tissues (tiles) were subsequently selected by a Swin-T-based tissue-type classifier. Up to 500 tumor tiles were randomly selected for each slide. Five NLP-based models (in orange) and four CNN-based models (in blue) were trained to predict tile-level biomarkers. Models with a red star represent models that are applied in digital pathology the first time. The predictive slide labels were obtained via tile score aggregation.

## Requirements

We used two environments to implement our paper due to the different versions of timm for *MobileViT* and *other models*

**Models Without MobileViT**

- ``python 3.8``
- ``torch 1.8.1+cu111``
- ``torchvision 0.9.1+cu111``
- ``timm 0.5.4``

**MobileViT**

- ``python 3.8``
- ``torch 1.8.1+cu111``
- ``torchvision 0.9.1+cu111``
- ``timm 0.6.8``


## Code

Firstly shoud:

```bash
cd NLP-based-digital-pathology
```

**Extract Tiles**

We extracted tiles by **extractTiles.py**. The origin of this code is from [kather lab](https://github.com/KatherLab/preProcessing).

```bash
python extractTiles.py -s slide_path -o out_path -ps pic_save_path
```

**Tissue Classification**

(Swin-T)-based Tissue classifier can be trained by from [our lab](https://github.com/Boomwwe/SOTA_MSI_prediction).

```bash
python Tissue_classfier.py -tr train_path -te test_path -ps model_save_path 
```

**Tile-Level Label Training**

Take MSI status prediction as an example. BRAF muation and CIMP status are similar as MSI (Only should change the ground truth file path in patch_dataloader.py). 

Parameters of models pretrained on ImageNet and code of [Sequecner2D module](https://arxiv.org/abs/2205.01972](https://github.com/okojoalg/sequencer) can be dwonloaded from baiduyun:链接: https://pan.baidu.com/s/1sHCx929L6KltFi5FsuOL9Q 提取码: 7buf

When training models without MobileViT

```bash
python MSI/train/train_external.py --TCGA_folder_path test_folder_path\
                                   --MCO_MSS_path train_folder_path_with_MSS\
                                   --MCO_MSI_path train_folder_path_with_MSI\
                                   --model_name model_name\
                                   --output_dir output_folder\
                                   --model_save_path model_save_path
```

When training MobileViT, you should change the python environment and train:

```bash
python MSI/train/train_external_mobilevit.py --TCGA_folder_path test_folder_path\
                                   --MCO_MSS_path train_folder_path_with_MSS\
                                   --MCO_MSI_path train_folder_path_with_MSI\
                                   --model_name model_name\
                                   --output_dir output_folder\
                                   --model_save_path model_save_path
```


**Patient-Level Label Prediction**

```bash
python MSI/pred/patient_pred.py --TCGA_folder_path prediction_folder_path\
                                --model_name model_name\
                                --model_save_path model_save_path\
                                --output_dir output_folder
```

Like training, when prediction by MobieViT:

```bash
python MSI/pred/patient_pred_mobilevit.py --TCGA_folder_path prediction_folder_path\
                                --model_name model_name\
                                --model_save_path model_save_path\
                                --output_dir output_folder
```
