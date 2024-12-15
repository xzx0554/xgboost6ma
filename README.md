# xgboost_6ma
Code Implementation for 'xgboost6ma' Paper

Given the constraints of limited memory and GPU memory, to execute 'xgboost6ma', it is necessary to first perform feature extraction using the associated DeepModel. The features extracted should be stored in `.pkl` files. Subsequently, you can proceed to train and evaluate the XGBoost classifier using the script `run_xgboost.py`.

Additionally, the deep learning checkpoints required for this process need to be downloaded from the working directories of `bert6ma` and `cnn6ma`, and placed into a folder named `model`.

kuratahiroyuki/CNN6mA
kuratahiroyuki/BERT6mA
