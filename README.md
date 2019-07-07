## Forked from

- [origin](https://mekhub.cn/danyabo/superconductor/tree/master)


## Benchmark Datasets for Superconductor Tc Prediction

- [DataH](https://mekhub.cn/danyabo/superconductor/blob/master/Dataset/DataH.csv)
- [DataV](https://mekhub.cn/danyabo/superconductor/blob/master/Dataset/DataV.csv)
- [DataK](https://mekhub.cn/danyabo/superconductor/blob/master/Dataset/DataK.csv)


## How to run
Train: python Conv_feature_extract.py --data Dataset/DataK.csv


Test: python ConvGBDT.py --data Dataset/DataV.csv --model_name check_point/DataV/best_model.ckpt.meta --model_path check_point/model_0
