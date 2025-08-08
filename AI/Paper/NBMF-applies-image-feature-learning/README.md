# 【NBMF应用图像特征学习】问题

## 课题简介
这是一个通用的论文复现项目模板，适用于各种领域的论文复现工作。本项目遵循量子开发实验室的代码规范和项目结构要求。

## 项目结构
```
NBMF-applies-image-feature-learning/            # 项目目录
├── README.md                                   # 项目说明文档
├── requirements.txt                            # 依赖包列表
├── params.py                                   # 参数文件
├── data_pretreat.py                            # 数据预处理文件
└── main.py                                     # 主程序入口
```

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行示例：

​	首次运行需加后缀“--pretreat”以执行数据预处理：

```bash
python main.py --mode=anneal_bcd --pretreat=true
```

​	数据预处理后会生成matrix_v.npy及matrix_v.csv文件，之后通过后缀“--mode”控制进入对应的求解算法函数，成功运行完成后会生成matrix_w.npy及matrix_w.csv文件，同时也会生成matrix_h.npy及matrix_h.csv文件，之后代码会根据保存文件进入数据后处理：

```bash
python main.py --mode=gurobi_bcd
```

​	若需要再次利用矩阵文件执行数据后处理而不进入算法求解过程，不加后缀直接执行即可：

```bash
python main.py
```

​	代码各函数参数可在参数文件params.py中进行统一调整。

## 算法说明

​	基于投影梯度法的交替非负最小二乘子问题算法：

- 算法目标：

  ​	算法的核心在于高效求解子问题：
  $$
  W^{k+1}=arg \min_{W \ge 0}\|V-XH^k\|_F
  $$
  ​	该子问题是带有边界约束的问题，需要采用投影梯度法来解决。将目标函数重写为向量子问题形式：
  $$
  f(H) = \frac{1}{2}\|V-WH\|_F^2 = \frac{1}{2}H_i^T\begin{bmatrix}
   W^TW & & \\
   & \ddots & \\
   & & W^TW
  \end{bmatrix}H_i + \text{$H$'s linear terms}
  $$
  

- 算法原理：

  ​	每个子问题都需要一个迭代过程，这些迭代被称为子迭代。在使用子迭代解决子问题时，必须在每个子迭代中保持梯度：
  $$
  \nabla f(H) = W^T(WH-V)
  $$
  ​	常数矩阵$W^TW$和$W^TV$可以分别在$O(nr²)$和$O(nmr)$操作时间内计算，然后进行子迭代。每次子迭代的主要计算任务是找到一个步长$\alpha$，使得**充分下降条件**：
  $$
  f(x^{k+1}) - f(x^{k}) \le \sigma \nabla f(x^{k})(x^{k+1}-x^{k})
  $$
  ​	得到满足。假设$H$为当前的解，则新解为：
  $$
  \hat H \equiv P[H - \alpha\nabla f(H)]
  $$
  ​	其中（$u_i$和$l_i$是上下界）：
  $$
  P[x_i]=\left\{\begin{matrix}
   x_i & \text{if $l_i \le x_i\le u_i$,}\\
   u_i & \text{if $x_i\ge u_i$,}\\
   l_i & \text{if $x_i\le l_i$,}
  \end{matrix}\right.
  $$
  ​	为了验证上面新解是否满足充分下降条件，计算需要$O(nmr)$次运算。若对$\hat H$进行$t$次试验，计算成本$O(tnmr)$将变得难以承受。为此，针对二次函数$f(x)$及任意向量$d$，提出以下策略以降低计算成本：
  $$
  f(x+d) = f(x) + \nabla f(x)^Td + d^T \nabla^2 f(x)d
  $$
  ​	因此，对于连续两次迭代$x^{k}$和$x^{k+1}$，充分下降条件可表示为：
  $$
  (1-\sigma) \nabla f(x^{k})^T(x^{k+1}-x^{k})+\frac{1}{2}(x^{k+1}-x^{k})^T \nabla^2 f(x^{k}) (x^{k+1}-x^{k}) \le 0
  $$
  ​	现在，根据子问题形式目标函数定义的$f(H)$是二次函数，因此充分下降条件变为：
  $$
  (1-\sigma) \left \langle \nabla f(H), \hat H - H \right \rangle + \frac{1}{2} \left \langle \hat H - H, (W^TW)(\hat H - H) \right \rangle \le 0
  $$
  ​	其中$\left \langle \cdot , \cdot \right \rangle$表示两个矩阵的分量积之和。

- 实现细节

  ​	实际过程中，对于子问题内层循环，最多尝试 20 次检查充分下降条件，每次根据充分下降条件判断是否需要减小步长，如果不满足则减小步长为原来的$\beta$份，否则将步长增大为原来的$\frac{1}{\beta}$ 倍，更新$H$并退出内层循环。

- 关键步骤

  ​	**停止条件的判断**既需要基于运行时间，又需要基于容差。如果投影梯度范数与初始梯度范数的比例小于容差，则停止迭代。如果交替非负最小二乘子问题函数只迭代了一次，说明容差过大导致迭代不充分，则将$W$梯度的容差缩小为原来的 0.1 倍。

- 性能分析

  ​	上述公式中的主要操作是矩阵乘法$(W^TW) \cdot (\hat H − H)$，其时间复杂度为$O(mr²)$。因此，检查的成本从$O(tnmr)$显著降低到$O(tmr²)$。考虑到计算$W^TV$的初始成本为$O(nmr)$，则解决子问题的时间复杂度为$O(nmr) + \text{sub-iterations} \times O(tmr^2)$，其中$t$表示每次子迭代的对充分下降条件的平均检查次数。

## 作者信息
- 作者姓名：周澍锦
- 联系方式：zhoushujin@tongji.edu.cn