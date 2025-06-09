# Interpretable and Inclusive MRI-Based Neurodegenerative Disease Classification
The adoption of deep learning in clinical settings is limited by its function as “black boxes,” which lacks transparency and offers little insight into
how its decisions are made. Furthermore, the lack of diverse demographics in commonly used brain imaging datasets may introduce biases that negatively
affect model performance for underrepresented populations.

This repository contains our paper on introducing interpretability methods to inclusive deep-learning models for neurodegenerative disease classification,
specifically Alzheimer's disease (AD).

## Requirements

To install dependencies, navigate to this repository and create a virtual environment using conda, venv, etc. Then run the following:

```
pip install -r requirements.txt
```

## Data
We used the [BrainLat](https://www.nature.com/articles/s41597-023-02806-8) dataset, which provides brain imaging data from 780 individuals in Latin America, focusing on underrepresented populations. Previous literature
showed that most models targeting neurodegenerative disease classification focused on datasets from Europe, the US, and other high-income regions.

We then combined BrainLat with the OASIS dataset to explore the impact of mixing data across studies and demographics. OASIS provides a cross-sectional collection
of 416 individuals in various AD stages.

## Repository Structure
```
.
├── src/
│   ├── dataset.py        # Script for defining new MRIDataset
|   └── data_utils.py     # Scripts for data preprocessing (background cleaning, normalization, data augmentation)
|   └── train.py          # Script for training with any model (ViT, ResNet, or Inception)
├── figures/              # Project figures
└── 
```

## Preprocessing

### Skull-Stripping
Skull-stripping, which removes the skull and other non-brain tissues from the image, is a common first step for pre-processing brain images. Generally, this step
aids in increasing accuracy and adding standardization before further analysis.

We used SynthStrip locally with Docker to apply skull-stripping to the images provided from BrainLat.
More information on how to replicate this process can be found on the SynthStrip documentation, found [here](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).

### Slice Extraction + Normalization
Normalization images, scaling pixels to be between 0 and 1, standardizes the input values. These steps lead to faster training and increased generalization.
We first normalized the image data, then extracted middle slices from the axial view.

### Data Augmentation
Incorporating data augmentation adds regularization to model training, which prevents overfitting and therefore increases testing accuracy. We introduced a 25% probability of removing various patches in the images.

## Training
We tuned three different deep-learning models: ViT-B/16, ResNet-50, and Inception-V3, due to their high performance on the ImageNet benchmark.

## Interpretability
TracIn from the library Captum calculates the influence of a given training example on a given test example, representing the loss change for the given test example if the given training example were removed and the model were re-trained. We used TracIn from the library Captum to select the top 10 most influential training examples. We also used Grad-CAM to generate head maps visualizing these most influential image regions. 

For our code, see ``Interpretability.ipynb``

## Citations
The data used within this paper is provided by BrainLat:
```
@inproceedings{2023brainlat,
  title={The BrainLat project, a multimodal neuroimaging dataset of neurodegeneration from underrepresented backgrounds},
  author={Prado, P., Medel, V., Gonzalez-Gomez, R. et al.},
  booktitle={Scientific Data},
  volume={10},
  pages={889},
  year={2023},
  publisher={Nature},
  doi={10.1038/s41597-023-02806-8}
}
```
