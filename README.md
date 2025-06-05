# 1. 复原环境

可以直接 pixi 复原环境。

包比较少，也可以自行 pip 装上：

```toml
[pypi-dependencies]
torch = ">=2.7.0, <3"
pywavelets = ">=1.8.0, <2"
ptwt = ">=0.1.9, <0.2"
pandas = ">=2.2.3, <3"
scikit-learn = ">=1.6.1, <2"
wfdb = ">=4.3.0, <5"
pycm = ">=4.3, <5"
wandb = ">=0.19.11, <0.20"
```

装完登录一下 wandb ，后续可以看到训练过程和评估结果。

```bash
wandb login
```

# 2. 准备数据集

确保 data/ 下是以下结构：

```
data/
├── mit-bih-arrhythmia-database-1.0.0
│   ├── 100.atr
│   ├── 100.dat
│   ├── 100.hea
│   ├── 100.xws
...
```

**然后删除掉 `102-0.atr`，它会导致后续处理出错。**

接下来切分 train-val-test：

```bash
python utils/data_utils.py
```

data/ 下会生成以下文件：

```
data/
├── test.csv
├── train.csv
├── val.csv
```

也有其它数据集可用，具体见 `utils/data_utils.py` 中的 `rootdir` 变量。

```python
#rootdir = 'data/european-st-t-database-1.0.0'
rootdir = 'data/mit-bih-arrhythmia-database-1.0.0'
#rootdir = 'data/mit-bih-st-change-database-1.0.0'
#rootdir = 'data/sudden-cardiac-death-holter-database-1.0.0'
```

# 3. 训练与评估

```bash
python train.py
python evaluate.py
```

配置都在脚本内，例如更换模型

```python
# model = WTLSTM(bidirectional=bidirectional, level=3)
# model = CNNLSTMWithResidual(bidirectional=bidirectional)
model = CNNLSTM(bidirectional=bidirectional)
```

更换训练设备

```python
# device = torch.device("cpu")
device = torch.device("cuda:0")
```
