# RBM
Pattern Classification Project



## 实验环境

mac Sierra 10.12.4

python 2.7.10：

- numpy (1.12.1)
- scikit-learn (0.18.1)

目录结构：

- dataset
  - TEST：测试集
  - TRAIN：训练集
  - *.npz：数据预处理后的训练集和测试集
  - mnist.pkl.gz：用于验证实验
- src
  - preprocess.py：数据预处理
  - RBM.py：模型构建，训练，测试
- report
  - report.pdf：报告
  - result.txt：结果输出
- README.md：代码使用说明



## 运行说明

#### preprocess

直接运行 `preprocess.py` 会在 dataset 目录下生成 `*.npz` ，对应经过预处理的训练集和测试集

#### run

直接运行 `RBM.py` 会在 report 目录下生成 `result.txt` ，包括所有模型的结果（前11个报告）以及验证实验的结果（后3个报告）



