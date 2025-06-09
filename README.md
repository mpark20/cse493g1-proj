# Interpretable and Inclusive MRI-Based Neurodegenerative Disease Classification

The adoption of deep learning in clinical settings is limited by its function as “black boxes,” which lacks transparency and offers little insight into
how its decisions are made. Furthermore, the lack of diverse demographics in commonly used brain imaging datasets may introduce biases that negatively
affect model performance for underrepresented populations.

This repository contains our paper on introducing interpretability methods to inclusive deep-learning models for neurodegenerative disease classification,
specifically Alzheimer's disease (AD).

## Data

We used the BrainLat dataset, which provides brain imaging data from 780 individuals in Latin America, focusing on underrepresented populations. Previous literature
showed that most models targeting neurodegenerative disease classification focused on datasets from Europe, the US, and other high-income regions.

We then combined BrainLat with the OASIS dataset to _. OASIS provides a cross-sectional collection of 416 individuals in various AD stages.

## Repository Structure
```
.
├── src/
│   ├── 
|   └── 
├── 
└── figures/
```

## Preprocessing

### Skull-Stripping

Skull-stripping, which removes the skull and other non-brain tissues from the image, is a common first step for pre-processing brain images. Generally, this step
aids in increasing accuracy and adding standardization before further analysis.

We used SynthStrip locally with Docker to apply skull-stripping to the images provided from BrainLat.
More information on how to replicate this process can be found on the SynthStrip documentation, found [here](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).

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
