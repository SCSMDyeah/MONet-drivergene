# MONet:Cancer Driver Gene Identification Algorithm Based on Integrated Analysis of Multi-Omics Data and Network Models
The progression of cancer results from the accumulation of driver gene mutations, which confer a selective growth advantage to cells. Identifying cancer driver genes is crucial for understanding the molecular mechanisms of cancer, developing targeted therapies, and discovering biomarkers. We proposes a method based on multi-omics data and network model integration analysis, named MONet(cancer driver gene identification algorithm based on integrated analysis of Multi-Omics data and Network models), for identifying cancer driver genes. Initially, we employ two graph neural network algorithms on protein-protein interaction (PPI) networks to learn the feature vector representations of each gene. Subsequently, the feature vectors obtained from the two algorithms are concatenated to form new gene feature vectors. Finally, these new feature vectors are input into a multi-layer perceptron model (MLP) to perform the semi-supervised cancer driver gene identification task. MONet assigns a probability to each gene for being a cancer driver gene, subsequently selecting genes that appear in at least two PPI networks as candidate driver genes. Experimental results demonstrate that MONet exhibits superior performance and stability across different PPI networks, outperforming baseline models in terms of the area under the receiver operating characteristic curve and the area under the precision-recall curve.

# Overview
Here we provide the code implementation of MONet. MONet runs in a Python environment.
- MONet.py the implementation of MONet.
- utils.py functions used in MONet.
- StratifiedKFold.py used for k-fold cross-validation.

# Implementation
- Step 1: Download the data according to the instructions in file dataset.txt.
- Step2: Open MONet.py to run MONet.

Tips:If you want to validate the k-fold cross-validation, you can run StratifiedKFold.py.
