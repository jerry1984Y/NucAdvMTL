# NucAdvMTL

To address the challenge in shared and private feature separation among similar ligands, a method named NucAdvMTL is proposed based on adversarial multi-task learning. This method constructs a dual-layer encoding architecture that integrates feature-sharing and private modules within a multi-task learning network coupled with a generative adversarial learning framework, effectively separating shared and private features in binding patterns. It mitigates potential negative transfer issues that may arise during shared feature extraction when dealing with datasets of varying ligand sizes. Simultaneously, it significantly boosts the model's capability to capture private characteristics. Benchmarking experiments demonstrate that NucAdvMTL outperforms current state-of-the-art prediction methods.

# 1. Requirements
Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

ProtTrans (ProtT5-XL-UniRef50 model)

# 2 Datasets
We provided a total of four benchmark datasets, namely Nuc-798, Nuc-849, Nuc-1521, and Nuc-207. Among them, Nuc-798, Nuc-849, and Nuc-1521 each consist of five common nucleotide (ATP, ADP, AMP, GTP, GDP) binding proteins constructed at different times. Nuc-207 comprises five uncommon nucleotide (TMP, CTP, CMP, UTP, UMP) binding proteins.

# 3. How to use
## 3.1 Set up environment for ProtTrans
Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master
## 3.2 Extract features
Extract pLMs embedding: cd to the NucAdvMTL/Feature_Extract dictionary, 
and run "python3 extract_prot.py", the pLMs embedding matrixs will be extracted to Dataset/prot_embedding folder.
## 3.3 Train and test
cd to the NucAdvMTL home dictionary.  
run "python3 NucAdvMTL.py" for training and testing the model for ATP, ADP, AMP, GTP, GDP binding residues prediction.  
