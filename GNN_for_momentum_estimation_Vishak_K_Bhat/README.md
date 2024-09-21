# Graph Neural Networks for Particle Momentum Estimation in the CMS Trigger System

## Previous works

Shravan Chaudhari 


 -> Boosted Top dataset was used which was also used in the paper https://arxiv.org/abs/2104.14659

 
 -> https://github.com/Shra1-25/GSoC2021_BoostedTopJets/tree/main All the scripts are here

Emre Kurtoglu


 -> https://medium.com/@emre.kurt.96/gsoc-2021-graph-neural-networks-for-particle-momentum-estimation-in-the-cms-trigger-system-2216e4e4d005

## Overview of the project

In physics, momentum is a physical quantity describing a particle’s motion, calculated as the **product of its mass and velocity**. It is a fundamental quantity, essential for describing
the motion and interactions of objects and particles. A muon particle is a subatomic particle similar to an electron, but with a much greater mass (about 207 times heavier). It is a type of lepton that participates in electromagnetic and weak interactions, but not in strong nuclear forces.


The Compact Muon Solenoid (CMS) is a detector situated at the Large Hadron Collider (LHC) close to Geneva, Switzerland. This experiment identifies the particles produced from collisions and analyzes their kinematics through the coordinated effort of various sub-detectors.


High momentum muons are an important object in numerous physics analyses at Compact Muon Solenoid(CMS), making the precise calculation of muon momentum essential. As mentioned in the [paper](https://iopscience.iop.org/article/10.1088/1742-6596/1085/4/042042), accurately distinguishing high momentum (signal) muons from the overwhelming low momentum (background) muons is crucial for the efficiency of the EMTF trigger. This distinction significantly lowers background rates, enhancing the trigger's effectiveness in identifying events of interest.


A muon traveling through the endcap detectors has a chance to leave hits in four sequential stations labeled 1, 2, 3, and 4. The specific combination of hits is called the mode. For each detector we have **7 features, namely, Phi coordinate, Theta coordinate, Bending angle, Time info, Ring number, Front/rear hit and Mask**. In addition, we have 3 more road variables: pattern straightness, zone, median theta. Thus, in total we have 7x4+3 = 31 features for each event.


Using this information we have to estimate the momentum of the particle. Initially discretized boosted decision tree was used followed by Fully Connected Neural Networks (FCNNs) and Convolutional Neural Networks (CNNs). In this repo we try to explore more about the Graph Neural Networks(GNN) to estimate the momentum of the muon particles.

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

## Datasets

The dataset consists of 1179356 muon events generated using Pythia and could be downloaded from the following [link](https://www.kaggle.com/datasets/ekurtoglu/cms-dataset)


## [Graph Dataset Creation](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Graph_creation)

- [x] **Each station as a node :** 
- [x] **Each feature as a node :**
- [x] **BendingAngle as a node :**
- [x] **Eta Value as a node :**


#### 1. [Each station as a node](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Graph_creation/Converting_to_graphs_eachstation_node.ipynb):
![APPROACH2png](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/assets/102585626/ae9ea64e-a429-4a0e-a1b1-518868fc0558)

**Here basically each station(4),is made a node of the graph and the node feature is the features recieved at the respective station(7)**
    
    Here,
         Total nodes = 4
         node feature length = 7
         edges should be between these 4 nodes (Yet to find out the best way to decide across which two nodes there should be and edge)


#### 2. [Each feature as a node](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Graph_creation/Converting_to_graphs_eachfeature_node.ipynb):
![APPROACH1](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/assets/102585626/1f295249-6a9f-4e28-9b2e-0ddb6d6a7e0e)


**Here basically each feature obtianed from different stations, here 4 stations are made a node of the graph and the node feature is the values of this feature across the 4 different stations**

    Here,
       Total nodes = 7
       node feature length = 4
       edges should be between these 7 nodes (Yet to find out the best way to decide across which two nodes there should be and edge)

#### 3. [Bending Angle as a Node](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/bendingAngle_node/A1/readme.md):

After experimenting the above 2 methods even though we were getting good results in terms of the MSE/MAE, the number of parameters in the GNN was 10 times that of the tabnet which we used as the baseline

So, we performed a lot of feature engineering and got these observations

- Mask values, frontValue, Time info, Ring number, Front/rear hit features were all the same across all the particles and had very less weightage in determining the momentum of the particle. Hence we dropped these features.
- Instead of using just phi angle as the feature we noticed that using sin(phi) and cos(phi) made more sense.
- Instead of using theta angle as a feature, eta value made more sense.
- After all this feature engineering, we had better featuers now to make our graph.
- Here we are using the bending angle to be the node feature, the sin(phi), cos(phi) and eta values as the edge features, and constructed a fully connected graph. 

![image](https://github.com/user-attachments/assets/84e22d1b-254e-4ac1-a986-0225deb06888)



#### 4. [Eta Value as a Node](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/etaValue_node/A2/readme.md):

- Here we are using the eta value to be the node feature, the sin(phi), cos(phi) and bending values as the edge features, and constructed a fully connected graph. 



## Experiment Results for 1 and 2

| Expt_NO | approach_of_graph  | Edges                    | Model                | Loss_fnc | loss | MAE  | Accuracy | F1 Score | Notebook |
|---------|---------------------|-----------------------|----------------------|----------|------|------|----------|----------|----------|
| 1       | each_station        | 0-1-2-3                  | 4 GCN                | pT Loss  | 3.83 | 43.21| 80.18    | 0.167    | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachstation_node/v2_A1/eachstation_node_v2.ipynb)     |
| 2       | each_station        | fully connected          | 4 MPL(embed-128)     | pT Loss  | 1.364| 15.82| 96.02    | 0.5529   | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachstation_node/v3_A1/eachstation_node_v3.ipynb)     |
| 3       | each_station        | fully connected          | 4 MPL(embed-64)      | pT Loss  | 1.3384|	16.22|	95.8146|	0.5422  | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachstation_node/v4_A1/eachstation_node_v4.ipynb)     |
| 4       | each_feature        | 0-1-2-3 // 2- (0,4, 5, 6)| 4 GCN                | pT Loss  | 4.38 | 45.33| 80.81    | 0.14     | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachfeature_node/v1_A2/eachfeature_node_v1.ipynb)     |
| 5       | each_feature        | fully connected          | 4 MPL                | pT Loss  | 3.7945|	41.7085|	80.337|	0.1682    | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachfeature_node/v2_A2/eachfeature_node_v2.ipynb)     |
| 6       | each_station        | fully connected          | 4 MPL(embed-64)      | MSE Loss  |0.000478|	52.64|	62.1498|	0.16685  | [link](https://www.kaggle.com/code/vishakkbhat/eachstation-node-v6/notebook)     |
| 7       | each_station        | fully connected          | 4 MPL(embed-64)      | MSE Loss  | 0.00263|	13.529|	97.458|	0.0515  | [link](https://www.kaggle.com/code/vishakkbhat/eachstation-node-v7/notebook)     |
| 8       | each_station        | 0-1-2-3                  | 4 MPL(embed-64)      | MSE Loss  | 0.00191|	13.3|	97.4053|	0.000326  | [link]()     |
| 9       | each_station        | fully connected          | 4 MPL(embed-64)      | pT Loss  | 0.004164|	2.0716|	100|	0.0  | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachstation_node/v9_A1/Eachstation_node.ipynb)     |
| 11       | each_station        | fully connected          | 4 MPL(embed-64)      | Custom Loss  | 0.46798|	0.8178|	100|	0.0  | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachstation_node/v11_A1/Eachstation_node_v11.ipynb)     |
| 12       | each_station        | fully connected          | 4 MPL(embed-64)      | Custom Loss  | 0.4538|	0.7849|	100|	0.0  | [link](https://github.com/Vishak-Bhat30/GSoC_24_GNN_For_Momentum/blob/main/Models/Eachstation_node/v12_A1/eachstation_node_v12.ipynb)     |


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

We can clearly see that GNN when eta value was taken as the only node feature and sin(phi), cos(phi), and bending angle taken as the edge features, we end up getting a better MAE and MSE compared to the tabnet model with a similar number of paramters. For the specific details regarding what each model is, you can refer to the code here.

Hence we can say that GNN have the capability to extract the information of the muon particles to estimate the momentum accurately.


