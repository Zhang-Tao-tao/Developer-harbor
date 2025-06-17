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

   ```bash
   cd cim
   python eval_0.py
   tar -cvzf mat.tar.gz mat
   ```

   下载打包好的CSV矩阵文件，并上传到[QBoson云平台](https://platform.qboson.com/)使用相干光量子计算机进行计算。

   运行结束后，得到打包好的报告文件和结果文件，可以使用rename.py文件进行重命名。

   ```bash
   python eval_1.py
   cd ..
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

#### optimizing-attention模型训练

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


</div>

2. 实验性能展示及与原始注意力模型对比结果：
   
   a) optimizing-attention模型训练的准确率和损失曲线图像如下：

   <img src="./images/model_acc_loss.png" width="1000" height="500">

   b) original-attention模型训练的准确率和损失曲线图像如下：

   <img src="./images/original_attn_a_l.png" width="1000" height="500">

3. 结果分析:

原始注意力模型在MNIST数据集上表现优异，准确率达到92.1%。而模型训练的结果表明optimizing-attention模型在MNIST数据集上表现良好，损失迅速下降，且准确率仅达到82.7%。由此说明，optimizing-attention模型在图像分类准确率上没有优势。




#### 仿真实验对比分析
1. 仿真实验对比结果
   我们通过在optimizing-attention模型上进行admm求解器和kaiwu模拟退火求解器的仿真对比实验，根据不同label计算置信度，从而绘制了一幅箱线图，如下所示：

   <img src="./images/boxplot_sa_vs_admm.png" width="1000" height="500">

2. 结果分析：
   1. 对于同一个label，大多数情况下optimizing-attention模型在admm求解器与kaiwu模拟退火表现相当。仅在label为3时，admm求解器的label置信度结果存在更大的波动。这说明kaiwu模拟退火求解器在求解时更加稳定。
   2. 对于不同的label，模型在判别label为0，1，4，6时具有较高的中位数和较窄的四分位距，这说明模型在判别以上label时具有相当高的置信度，而对于label=5存在比较严重的误判情况，因为此时的中位数较低且分布范围广。该结论对两种求解器均成立。这说明该模型对于label=5的判别效果较差。



#### QBoson CPQC-1 CIM 量子退火真机推理


#### 稀疏性分析


- 总结讨论

   <span style="color:blue">各种各样乱七八糟的问题(DP)</span>

## 作者信息
- 作者姓名：[请填写作者姓名]