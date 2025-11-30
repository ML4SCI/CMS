# Graph Neural Network Project

## Overview

This project involves training and evaluating a Graph Neural Network (GNN) model for predicting particle momentum. It includes scripts for data preparation, training, testing, and evaluating the model, as well as plotting results and saving them.

We noticed and came up with these observations:

1) Instead of taking eachStation as a node or each feature as a node, its better to take only the bendingAngle for each station to be the node feature. Also further we noticed that taking eta values as the node features gave better results compared to using bending values.

2) Talking about the Edge feature we introduced the 3 dimensional vector feature. Which includes sin(phi), cos(phi), and eta (-log(tan(theta/2)))

3) Using these node and edge features we came up with a message passign layers with different number of hidden layers to fit the data and we see the results in the results section.

## Files

- **dataset.py**: Contains data preparation and dataset handling functions.
- **losses.py**: Defines custom loss functions used for training the model.
- **main.py**: Main script for training the model, evaluating it, and generating plots.
- **models.py**: Defines the GNN model architecture.
- **train.py**: Script specifically for training the model.
- **utils.py**: Includes utility functions for training, evaluation, and plotting.
- **requirements.txt**: Lists the dependencies required for the project.
- **readme.md**: This README file.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum.git
   cd ./GSoC_24_GNN_For_Momentum/modular_code
2. **Create an environment**:
   ```bash
   conda create -n gsoc
   conda activate gsoc
   pip install -r requirements.txt
## Usage

1. **Using bending angle as node feature**
   ```bash
   python main.py --input_csv "/home/azureuser/vishak/GNN_KNN/CMS_trigger.csv" \
                  --batch_size 32 \
                  --learning_rate 0.001 \
                  --node_feat "bendAngle" \
                  --epochs 18 \
                  --save_dir "./outputs"

2. **Using etaValue as node feature**
   ```bash
   python main.py --input_csv "/home/azureuser/vishak/GNN_KNN/CMS_trigger.csv" \
                  --batch_size 32 \
                  --learning_rate 0.001 \
                  --node_feat "etaValue" \
                  --epochs 18 \
                  --save_dir "./outputs"



## Experiment Results for 3 and 4

| **Model**            | **MAE**   | **MSE**   | **Avg Inference Time (μs)** | **No. of Parameters** | **Model Code Link**         |
|----------------------|-----------|-----------|------------------------------|-----------------------|------------------------------|
| **TabNet**           | 0.9607    | 2.9746    | 458.7                        | 6696                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/baseline_tabnet/tabnet.ipynb)  |
| GNN-bendAngle    | 1.202931  | 3.520059  | 522.204                      | 5579                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/bendingAngle_node/A1/bendingAngle_node.ipynb)  |
| GNN-bendAngle2   | 1.215189  | 3.605358  | 530.19                            | 5903                  | [Link](url)  |
| GNN-etaValue    | 1.146910  | 3.240220  | 522.204                      | 5579                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/etaValue_node/A1/eta_value_1.ipynb)  |
| **GNN-etaValue2**    | 0.992087  | 2.525927  | 530.19                            | 5903                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/etaValue_node/A2/eta_value_2.ipynb)  |
| GNN-etaValue3    | 1.145697         | 3.276628         | 614.565                            | 6437                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/etaValue_node/A3/eta_value_3.ipynb)  |
| GNN-etaValue4    | 1.133285         | 3.205457         | 267.509                             | 6112                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/etaValue_node/A4/eta_value_4.ipynb)  |
| GNN-etaValue5    | 0.941620  | 2.312492  | 515.472                      | 6545                  | [Link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/etaValue_node/A5/eta_value_5.ipynb)  |

