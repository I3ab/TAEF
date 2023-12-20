# TAEF
we provide a code for TAEF: Transformer-Based Autoencoder Framework for Nonlinear Hyperspectral Anomaly Detection Detection. (submitted to TGRS)

# running code
python main_TAEF.py

# required changes when running code
line 19 in main_TAEF: path
line 163 in main_TAEF: path

# dataset
we provide two datasets for verifying: pavia, MUUFL.

# required changes when changing dataset
line 17 in main_TAEF: loadData
line 75 in main_TAEF: stop loss
line 62 in net_TAEF: net setting

# note
we have provide the output of the preprocessing and the patch generation method: pavia_split.mat 
,and the main_TAEF can use it directly. For other datasets, you can change the line 5 and line 93 of preprocessing_patch.m to generate the corresponding output. 
