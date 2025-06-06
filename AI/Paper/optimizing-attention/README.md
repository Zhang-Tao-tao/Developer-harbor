# 论文复现项目模板

## 模板简介
这是一个通用的论文复现项目模板，适用于各种领域的论文复现工作。本项目遵循量子开发实验室的代码规范和项目结构要求。

## 模板结构
```
paper-demo/              # 项目模板目录
├── README.md           # 项目说明文档
├── requirements.txt    # 依赖包列表
└── demo.py            # 主程序入口
```

## 使用方法
1. 安装依赖：

   - 切换工作目录到experiments

      ```bash
      cd path/to/project/AI/Paper/optimizing-attention/experiments/
      ```

   - 创建conda环境

      ```bash
      conda create -n optimizing-attention python==3.8.10 -y
      conda activate optimizing-attention
      ```

   - 安装子模块lqp_py用于admm求解器

      ```bash
      git submodule init .
      git submodule update . 
      
      cd admm
      pip install -e . 
      cd ..
      ```

   - 安装python依赖库

      ```bash
      pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
      pip install einops line_profiler icecream tqdm
      ```
   
   - 安装kaiwu库

      从[QBoson平台](https://platform.qboson.com/sdkDownload)下载安装kaiwuSDK并解压。

      ```bash
      pip install kaiwu-1.1.2-py3-none-any.whl 
      ```



2. 训练示例：

   ```bash
   python train.py
   ```

3. 模拟退火推理示例：

   ```bash
   python eval.py
   ```

4. CIM光量子计算机推理实例

   <span style="color:red">TODO</span>

   ```bash
   python cim.py --image_path xxx.png
   ```

## 算法说明
- 算法原理

   <span style="color:red">这里写详细的算法原理(WSL)</span>

- 实现细节

   <span style="color:red">这里写算法实现的流程，模型流程(DP)</span>

- 关键步骤

   <span style="color:blue">这里简要讲解整个文章的流程，分条列举文章的优势，为什么是优势(WSL)</span>

- 创新点

   <span style="color:blue">论文创新点(DP)</span>

### 性能分析

   <span style="color:blue">实验(LSC)</span>

#### 模型训练

1. 实验配置:

- 数据集：MNIST数据集中的70000张图片，按照6：1划分训练集和测试集

- 实验设备：NVIDIA GeForce RTX 4090 



- 训练参数：

<div align="center">

| **参数名**         | **值**             |
|:-------------------:|:-------------------:|
| num_epoch          | 50                 |
| batch_size         | 128                |
| learning_rate      | 0.001              |
| optimizer          | Adam               |
| loss_function      | CrossEntropyLoss   |

</div>

- 模型参数配置：

<div align="center">

| **参数名**     | **值**            |
|:--------------:|:----------------:|
| image_size     | (28, 28)         |
| patch_size     | (4, 4)           |
| num_classes    | 10               |
| dim            | 64               |
| channels       | 1                |
| dropout        | 0.0              |
| emb_dropout    | 0.0              |

</div>

2. 实验性能：


模型训练的准确率和损失曲线图像如下：

<img src="./images/model_loss_acc.png" width="1200" height="800">


3. 结果分析:

模型训练的结果表明optimizing-attention模型在MNIST数据集上表现良好，损失迅速下降，且准确率达到88.5%以上。




#### kaiwuSDK模拟退火推理



#### QBoson CPQC-1 CIM 量子退火真机推理


#### 稀疏性分析


- 总结讨论

   <span style="color:blue">各种各样乱七八糟的问题(DP)</span>

## 作者信息
- 作者姓名：[请填写作者姓名]