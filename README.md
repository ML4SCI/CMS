# CMS

![GitHub Logo](images/CollisionImage.png)

This project aims to improve machine learning algorithms for applications related to the Compact Muon Solenoid experiment at the Large Hadron Collider. The CMS algorithm is one of the largest particle collision experiments in the world at very high collision energies allowing rare insight into fundamental particle physics.

# Directory Structure

```
|-Projects
|---E2E - End-to-End event reconstruction and classification
|---GNNs - Graph Neural Networks for fast algorithms at CMS including momentum estimation and event triggers.
|---MAEs - Masked autoencoder pretraining to improve offline CMS analyses.
```

## E2E
End-to-End event regression and classification models rely on low-level, minimally processed data in formats such as images or point clouds to directly predict properties of a collision event.

### Previous contributions are included below:

| Project Title | Author | Year |
| :------------ | ------ | ---: |
| E2E DeepLearning | Purva Chaudhari | 2021 |

Note: more recent E2E projects have been moved to https://github.com/ML4SCI/E2E

## GNNs
Graph Neural Networks allow for very fast computation through sparsified operations that can also account for adjacency and other enforced symmetries. These are ideal for real-time calculation of object physics at the CMS experiment.

### Previous contributions are included below:
| Project Title | Author | Year |
| :------------ | ------ | ---: |
| GNN for Momentum Estimation | Emre Kurtoglu | 2021 |
| GNN for Momentum Estimation | Vishak K. Bhat| 2024 |

Note: Emre's code can be found at:
https://github.com/ekurtgl/GSoC-2021-GNN_for_Trigger

## MAEs
Masked autoencoder pretraining has been shown to improve reconstruction and classification neural networks. This project aims to extend these techniques to CMS data using physics informed pretraining methods and new and existing models.

| Project Title | Author | Year |
| :------------ | ------ | ---: |
| Particle Masking Autoencoder | Eric Reinhardt | 2023 |
| Hybrid ParT and L-GATr Transformers | Thanh "James" Nguyen | 2025 |