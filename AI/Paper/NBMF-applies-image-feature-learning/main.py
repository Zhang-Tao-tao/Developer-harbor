"""
NBMF Applies Image Feature Learning

这个文件是本项目的主函数。

作者：周澍锦
联系方式：zhoushujin@tongji.edu.cn
"""

import os
import json
import argparse
import numpy as np
from numpy.linalg import norm
from time import time
from PIL import Image
import matplotlib.pyplot as plt

from scipy.optimize import nnls
from scipy.optimize import minimize
from sklearn.decomposition import NMF
import kaiwu as kw
from gurobipy import Model, GRB
import gurobipy as gp

import params
import data_pretreat


# 授权初始化代码
# 示例的user_id和sdk_code无效，需要替换成自己的用户ID和SDK授权码
kw.license.init(user_id="123456", sdk_code="AjUvlTvWrWZoeidADu5Vbf6pceVmuX")

file_list_path = params.file_list_path
result_list_path = params.result_list_path
magic_number = params.magic_number
width = params.width
height = params.height
max_gray_value = params.max_gray_value
v_select = params.v_select
columns = params.columns
save_dir_path = params.save_dir_path
machine_name = params.machine_name


def construct_qubo(vi, w):
    """
    构造QUBO系数矩阵

    Args:
        vi: int * int, 整体矩阵列向量
        w: float * float, 宽度分解矩阵

    Returns:
        qubo: int * int, QUBO系数矩阵
    """
    n, k = w.shape[0], w.shape[1]
    # 构建QUBO矩阵，注意满足上三角矩阵的条件
    qubo = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            qubo[i][j] = 2 * w[:, i] @ w[:, j]

        qubo[i][i] = w[:, i] @ w[:, i] - 2 * w[:, i] @ vi

    return qubo


def solve_qubo_anneal(qubo, k):
    """
    调用kaiwu经典求解器.求解QUBO问题

    Args:
        qubo: int * int, QUBO系数矩阵
        k: int, 分解特征值

    Returns:
        res: int * int, 最优0-1向量
    """
    ising = kw.qubo.qubo_matrix_to_ising_matrix(qubo)
    solver = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,
                                                      alpha=0.99,
                                                      cutoff_temperature=0.001,
                                                      iterations_per_t=10,
                                                      size_limit=10)
    solution = solver.solve(ising[0])
    if solution.shape == (0,):
        return solution

    opt = kw.sampler.optimal_sampler(ising[0], solution, 0)

    if opt[0][0][k] == 1:
        res = opt[0][0][:k]
        res = (res + 1) / 2
    else:
        res = opt[0][0][:k]
        res = (-res + 1) / 2

    return res


def nls_sub_prob(v, w, h_init, tol, max_iter):
    """
    求解非负最小二乘子问题

    Args:
        v: int * int, 整体矩阵
        w: float * float, 宽度分解矩阵
        h_init: int * int, 初始高度分解矩阵
        tol: float, 停止阈值
        max_iter: int, 最大迭代次数

    Returns:
        h: int * int, 高度分解矩阵
        grad: int * int, 与高度分解矩阵同规模的梯度矩阵
        iteration: int, 迭代次数
    """
    # 初始化 H 为初始值，并计算矩阵的乘积
    h = h_init
    wtv = np.dot(w.T, v)
    wtw = np.dot(w.T, w)

    # 初始化步长 alpha 和步长衰减因子 beta
    alpha = 1
    beta = 0.1
    # 开始子问题的迭代，迭代次数从 1 到 max_iter
    for iteration in range(1, max_iter):
        # 计算梯度
        grad = np.dot(wtw, h) - wtv
        # 计算投影梯度的范数
        proj_grad = norm(grad[np.logical_or(grad < 0, h > 0)])
        # 如果投影梯度范数小于容差，则停止子问题的迭代
        if proj_grad < tol:
            break

        # 搜索步长
        # 内层循环，最多尝试 20 次
        for inner_iter in range(1, 20):
            # 计算新的 H 值
            hn = h - alpha * grad
            # 确保 Hn 中的所有元素非负
            hn = np.where(hn > 0, hn, 0)
            # 计算 H 的变化量
            d = hn - h
            # 计算梯度与变化量的点积
            grad_d = np.sum(grad * d)
            # 计算二次型 d^T * WtW * d
            dqd = np.sum(np.dot(wtw, d) * d)

            # 判断是否满足充分下降条件
            suff_decr = ((0.99 * grad_d + 0.5 * dqd) < 0)
            # 如果是第一次内层迭代，根据充分下降条件判断是否需要减小步长
            if inner_iter == 1:
                decr_alpha = not suff_decr
                hp = h
            # 如果需要减小步长
            if decr_alpha:
                # 如果满足充分下降条件，则更新 H 并退出内层循环
                if suff_decr:
                    h = hn
                    break
                # 否则，将步长缩小为原来的 beta 倍
                else:
                    alpha = alpha * beta
            # 如果不需要减小步长
            else:
                # 如果不满足充分下降条件或者 Hp 与 Hn 相等，则恢复 H 并退出内层循环
                if not suff_decr or (hp == hn).all():
                    h = hp
                    break
                # 否则，将步长增大为原来的 1/beta 倍，并更新 Hp
                else:
                    alpha = alpha / beta
                    hp = hn

        # 如果达到子问题迭代的最大次数，打印提示信息
        if iteration == max_iter:
            print('Max iter in nls_sub_prob')

    return h, grad, iteration


def nbmf_anneal_nsp(k, tol, time_limit, max_iter):
    """
    退火方式+非负最小二乘子问题实现nbmf

    Args:
        k: int, 分解特征值
        tol: float, 停止阈值
        time_limit: float, 时间限制
        max_iter: int, 最大迭代次数

    Returns:
        v: int * int, 整体矩阵
        w: float * float, 宽度分解矩阵
        h: int * int, 高度分解矩阵
    """
    global v_select, max_gray_value
    v_origin = np.load('matrix_v.npy')
    v = v_origin[:, v_select]
    n, m = v.shape[0], v.shape[1]
    # 初始化矩阵 W 和0-1矩阵 H 为初始值，并记录开始时间
    w = np.random.rand(n, k) * (max_gray_value + 0.5)
    h = np.random.randint(0, 2, (k, m))
    init_t = time()

    # 计算 W 和 H 的梯度
    grad_w = np.dot(w, np.dot(h, h.T)) - np.dot(v, h.T)
    grad_h = np.dot(np.dot(w.T, w), h) - np.dot(w.T, v)
    # 计算初始梯度的范数
    init_grad = norm(np.r_[grad_w, grad_h.T])
    print('Init gradient norm %f' % init_grad)
    # 计算 W 梯度的容差，取 0.001 和 tol 中的较大值乘以初始梯度范数
    tol_w = max(0.001, tol) * init_grad
    tol_h = tol_w

    for iteration in range(1, max_iter):
        # 停止条件判断
        # 计算投影梯度的范数
        proj_norm = norm(np.r_[grad_w[np.logical_or(grad_w < 0, w > 0)],
                               grad_h[np.logical_or(grad_h < 0, h > 0)]])
        # 如果投影梯度范数小于容差乘以初始梯度范数，或者运行时间超过时间限制，则停止迭代
        if proj_norm < tol * init_grad or time() - init_t > time_limit:
            break

        # 调用 nls_sub_prob 函数更新 W，并获取更新后的 W、梯度和迭代次数
        (w, grad_w, iter_w) = nls_sub_prob(v.T, h.T, w.T, tol_w, 1000)
        # 转置 W 和 grad_W 以恢复原始形状
        w = w.T
        grad_w = grad_w.T

        # 如果 nls_sub_prob 函数只迭代了一次，则将 W 梯度的容差缩小为原来的 0.1 倍
        if iter_w == 1:
            tol_w = 0.1 * tol_w

        # Step 2: 固定W，优化H（QUBO求解）
        for i in range(m):
            vi = v[:, i]
            qubo = construct_qubo(vi, w)
            hi = solve_qubo_anneal(qubo, k)
            if hi.shape != (0,):
                h[:, i] = hi

        # 每迭代 10 次，输出一个点表示进度
        if iteration % 10 == 0:
            # stdout.write('.')
            print('.')

    # 打印最终的迭代次数和投影梯度范数
    print('\nIter = %d Final proj-grad norm %f' % (iteration, proj_norm))
    return v, w, h


def block_coordinate_descent(v, w, h, alpha=0.1):
    """
    块坐标下降，使用非负最小二乘逐列求解添加正则化的矩阵分解（update_w_nnls）

    Args:
        v: int * int, 整体矩阵
        w: float * float, 宽度分解矩阵
        h: int * int, 高度分解矩阵
        alpha: float, 正则化参数

    Returns:
        w: float * float, 更新后的宽度分解矩阵
    """
    k = h.shape[0]
    # 添加正则化项
    a = np.vstack([h.T, np.sqrt(alpha) * np.eye(k)])
    for i in range(v.shape[0]):
        # 扩展目标向量
        b = np.concatenate([v[i], np.zeros(k)])
        # 求解NNLS，w[i]: (35, )
        w[i], _ = nnls(a, b)

    return w


def nbmf_anneal_bcd(k, max_iter=100, alpha=0.1):
    """
    退火方式+块坐标下降实现nbmf

    Args:
        k: int, 分解特征值
        max_iter: int, 最大迭代数
        alpha: float, 正则化参数

    Returns:
        v: int * int, 整体矩阵
        w: float * float, 宽度分解矩阵
        h: int * int, 高度分解矩阵
    """
    global v_select, max_gray_value
    v_origin = np.load('matrix_v.npy')
    v = v_origin[:, v_select]
    n, m = v.shape[0], v.shape[1]
    # 随机初始化二进制矩阵W和H
    w = np.random.rand(n, k) * (max_gray_value + 0.5)
    h = np.random.randint(0, 2, (k, m))

    for epoch in range(max_iter):
        # Step 1: 固定H，优化W（非负最小二乘 + 正则化）
        # 块坐标下降，使用非负最小二乘逐列求解（添加正则化）
        w = block_coordinate_descent(v, w, h, alpha=alpha)

        # Step 2: 固定W，优化H（QUBO求解）
        for i in range(m):
            vi = v[:, i]
            qubo = construct_qubo(vi, w)
            hi = solve_qubo_anneal(qubo, k)
            if hi.shape != (0,):
                h[:, i] = hi

    return v, w, h


def solve_qubo_gurobi(qubo, k, time_limit=None, verbose=False):
    """
    使用Gurobi求解求解QUBO问题

    Args:
        qubo: int * int, QUBO系数矩阵
        k: int, 分解特征值
        time_limit: float, 求解时间限制（秒）
        verbose: bool, 是否显示求解过程信息

    Returns:
        dict: {
            'solution': list, 最优解向量
            'objective': float, 目标函数值
            'runtime': float, 求解时间（秒）
            'status': str, 求解状态
        }
    """
    # 验证输入矩阵
    if qubo.shape != (k, k):
        raise ValueError("QUBO系数矩阵必须是k规模方阵")

    try:
        # 1. 创建模型
        model = gp.Model("QUBO")
        if not verbose:
            model.Params.OutputFlag = 0

        # 2. 创建二元变量
        x = model.addVars(k, vtype=GRB.BINARY, name="x")

        # 3. 构建目标函数
        objective = gp.QuadExpr()

        # 添加线性项（对角线元素）
        for i in range(k):
            objective += qubo[i, i] * x[i]

        # 添加二次项（上三角部分）
        for i in range(k):
            for j in range(i + 1, k):
                # 注意：qubo_matrix[i,j]对应x_i*x_j项的系数
                objective += qubo[i, j] * x[i] * x[j]

        # 设置求解目标方向
        model.setObjective(objective, GRB.MINIMIZE)

        # 4. 设置求解参数（时间限制）
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        # 5. 求解模型
        start_time = time()
        model.optimize()
        runtime = time() - start_time

        # 6. 处理结果
        if model.status == GRB.OPTIMAL:
            solution = [int(x[i].X) for i in range(k)]
            obj_value = model.ObjVal
            status = "OPTIMAL"
        elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
            solution = [int(x[i].X) for i in range(k)]
            obj_value = model.ObjVal
            status = "TIME_LIMIT (feasible solution found)"
        else:
            solution = None
            obj_value = None
            status = model.status

        return {
            'solution': solution,
            'objective': obj_value,
            'runtime': runtime,
            'status': status
        }

    except gp.GurobiError as e:
        print(f"Gurobi错误: {str(e)}")
        return {
            'solution': None,
            'objective': None,
            'runtime': 0,
            'status': f"ERROR: {str(e)}"
        }


def nbmf_gurobi_bcd(k, time_limit=1, max_iter=100, alpha=0.1):
    """
    Gurobi方式+块坐标下降实现nbmf

    Args:
        k: int, 分解特征值
        time_limit: float, 时间限制
        max_iter: int, 最大迭代数
        alpha: float, 正则化参数
        time_limit

    Returns:
        v: int * int, 整体矩阵
        w: float * float, 宽度分解矩阵
        h: int * int, 高度分解矩阵
    """
    global v_select, max_gray_value
    v_origin = np.load('matrix_v.npy')
    v = v_origin[:, v_select]
    n, m = v.shape[0], v.shape[1]
    # 随机初始化二进制矩阵W和H
    w = np.random.rand(n, k) * (max_gray_value + 0.5)
    h = np.random.randint(0, 2, (k, m))

    for epoch in range(max_iter):
        # Step 1: 固定H，优化W（非负最小二乘 + 正则化）
        # 块坐标下降，使用非负最小二乘逐列求解（添加正则化）
        w = block_coordinate_descent(v, w, h, alpha=alpha)

        # Step 2: 固定W，优化H（Gurobi求解）
        for i in range(m):
            vi = v[:, i]
            qubo = construct_qubo(vi, w)
            hi = solve_qubo_gurobi(qubo, k, time_limit, True)['solution']
            if hi is not None:
                h[:, i] = hi

    return v, w, h


def solve_qubo_cim(qubo, k, bit):
    """
    调用kaiwu经典求解器.求解QUBO问题

    Args:
        qubo: int * int, QUBO系数矩阵
        k: int, 分解特征值
        bit: int, 真机CIM指数比特数

    Returns:
        res: int * int, 最优0-1向量
    """
    global save_dir_path, machine_name
    # 根据比特数确定机器名
    assert ((bit == 100 and machine_name == "CPQC-100") \
            or (bit == 550 and machine_name == "CPQC-550")), "CIM真机名称错误"

    kw.common.CheckpointManager.save_dir = save_dir_path

    # 调整矩阵精度，否则精度校验会出现“CSV数据文件的精度过高”问题
    qubo_precision = kw.qubo.adjust_qubo_matrix_precision(qubo)

    ising = kw.qubo.qubo_matrix_to_ising_matrix(qubo_precision)
    optimizer = kw.cim.CIMOptimizer(user_id='1241241515', sdk_code='absd1232',
                                    task_name="test", machine_name=params.machine_name)
    solution = optimizer.solve(ising[0])
    opt = kw.sampler.optimal_sampler(ising[0], solution, 0)

    if opt[0][0][k] == 1:
        res = opt[0][0][:k]
        res = (res + 1) / 2
    else:
        res = opt[0][0][:k]
        res = (-res + 1) / 2

    return res


def nbmf_cim_bcd(bit, k, max_iter=100, alpha=0.1):
    """
    真机CIM+块坐标下降实现nbmf

    Args:
        bit: int, 真机CIM指数比特数
        k: int, 分解特征值
        max_iter: int, 最大迭代数
        alpha: float, 正则化参数

    Returns:
        v: int * int, 整体矩阵
        w: float * float, 宽度分解矩阵
        h: int * int, 高度分解矩阵
    """
    global v_select, max_gray_value

    # 根据比特数确定机器名
    assert ((bit == 100) or (bit == 550)), "CIM真机比特数错误"

    v_origin = np.load('matrix_v.npy')
    v = v_origin[:, v_select]
    n, m = v.shape[0], v.shape[1]
    # 随机初始化二进制矩阵W和H
    w = np.random.rand(n, k) * (max_gray_value + 0.5)
    h = np.random.randint(0, 2, (k, m))

    for epoch in range(max_iter):
        # Step 1: 固定H，优化W（非负最小二乘 + 正则化）
        # 块坐标下降，使用非负最小二乘逐列求解（添加正则化）
        w = block_coordinate_descent(v, w, h, alpha=alpha)

        # Step 2: 固定W，优化H（QUBO求解）
        i = 0
        mod = bit // k
        v_mod = np.zeros(n)
        w_mod = np.hstack((w,) * mod)
        while i < m:
            if (i + 1) % mod != 0:
                v_mod += v[:, i]
                i += 1
            else:
                v_mod += v[:, i]
                qubo = construct_qubo(v_mod, w_mod)
                h_mod = solve_qubo_cim(qubo, k, bit)
                for j in range(mod):
                    h[:, i + j + 1 - mod] = h_mod[(j * k):((j + 1) * k)]
                v_mod = np.zeros((n, 1))
                i += 1

    return v, w, h


def matrix_save(w, h):
    """
    保存分解矩阵

    Args:
        w: float * float, 宽度分解矩阵
        h: int * int, 高度分解矩阵

    Returns:
        None
    """
    np.save('matrix_w.npy', w)
    np.savetxt('matrix_w.csv', w, delimiter=',', fmt='%f')
    np.save('matrix_h.npy', h)
    np.savetxt('matrix_h.csv', h, delimiter=',', fmt='%f')

    return


def result_generate(column=0):
    """
    合成结果向量

    Args:
        column: int, 结果向量列数

    Returns:
        pixels: int * int, 像素矩阵
    """
    global width, height, max_gray_value

    # 读取矩阵数据
    w = np.load('matrix_w.npy')
    h = np.load('matrix_h.npy')

    # 边界判断
    assert 0 <= column < h.shape[1], "指定合成列数越界"

    # 将像素列表转换为NumPy数组
    wh = w @ h[:, column]
    wh = wh.astype(int)
    wh = np.clip(wh, 0, max_gray_value)
    pixels = wh.reshape(width, height)

    # 显示图像
    # plt.imshow(pixels, cmap='gray', vmin=0, vmax=max_gray_value)
    # plt.show()

    return pixels


def loss_calculate(pixels, column=0):
    """
    计算图像重构误差

    Args:
        pixels: int * int, 像素矩阵
        column: int, 结果向量列数

    Returns:
        None
    """
    global width, height, v_select

    # 读取矩阵数据
    v_origin = np.load('matrix_v.npy')
    v = v_origin[:, v_select]

    # 边界判断
    assert 0 <= column < v.shape[1], "指定计算列数越界"

    vec1 = pixels.reshape(width * height, -1)
    vec2 = v[:, column]

    # 计算向量间欧氏距离
    loss = np.linalg.norm(vec1 - vec2)
    print(f"The Loss of column {column} is {loss}.")

    return loss


def pgm_write_binary(pgm_path, pixels):
    """
    写入二进制pgm图像文件

    Args:
        pgm_path: string, pgm文件路径
        pixels: int * int, pgm文件像素向量

    Returns:
        None
    """
    global magic_number, width, height, max_gray_value

    # 根据文件编码方式（P5二进制）打开
    with open(pgm_path, 'wb') as f:
        # 写入头部信息，包括魔数、尺寸、最大灰度值等
        f.write(f"{magic_number}\n{width} {height}\n{max_gray_value}\n".encode())

        # 写入像素数据
        data = pixels.astype(np.uint8)
        f.write(data.tobytes())

        f.close()

    return


def data_postreat():
    """
    数据后处理

    Args:
        columns: list, 结果向量列数列表

    Returns:
        None
    """
    global result_list_path, columns
    for column in columns:
        pixels = result_generate(column)
        loss_calculate(pixels, column)
        pgm_write_binary(result_list_path + "output" + str(column) + ".pgm", pixels)

    return


def main():
    """
    主函数入口

    Args:
    
    Returns:
        None
    """
    # 参数处理
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretreat', default='false')
    parser.add_argument('--mode', default='default')
    ps = parser.parse_args()

    # 数据预处理
    if ps.pretreat == 'true':
        print('Pretreating Data...')
        data_pretreat.data_pretreat()

    # 退火NSP矩阵分解
    if ps.mode == 'anneal_nsp':
        start_time = time()
        print('Initializing Anneal_nsp Training Process...')
        v, w, h = nbmf_anneal_nsp(params.anneal_nsp_k,
                                  params.anneal_nsp_tol,
                                  params.anneal_nsp_time_limit,
                                  params.anneal_nsp_max_iter)
        runtime = time() - start_time
        print(f'Total runtime is {runtime}.')
        # 保存分解矩阵
        matrix_save(w, h)

    # 退火BCD矩阵分解
    if ps.mode == 'anneal_bcd':
        print('Initializing Anneal_bcd Training Process...')
        start_time = time()
        v, w, h = nbmf_anneal_bcd(params.anneal_bcd_k,
                                  params.anneal_bcd_max_iter,
                                  params.anneal_bcd_alpha)
        runtime = time() - start_time
        print(f'Total runtime is {runtime}.')
        # 保存分解矩阵
        matrix_save(w, h)

    # Gurobi+BCD矩阵分解
    if ps.mode == 'gurobi_bcd':
        print('Initializing Gurobi Training Process...')
        start_time = time()
        v, w, h = nbmf_gurobi_bcd(params.gurobi_bcd_k,
                                  params.gurobi_bcd_time_limit,
                                  params.gurobi_bcd_max_iter,
                                  params.gurobi_bcd_alpha)
        runtime = time() - start_time
        print(f'Total runtime is {runtime}.')
        # 保存分解矩阵
        matrix_save(w, h)

    # CIM真机BCD矩阵分解
    if ps.mode == 'cim_bcd':
        print('Initializing Anneal_bcd Training Process...')
        start_time = time()
        v, w, h = nbmf_cim_bcd(params.cim_bcd_bit,
                               params.cim_bcd_k,
                               params.cim_bcd_max_iter,
                               params.cim_bcd_alpha)
        runtime = time() - start_time
        print(f'Total runtime is {runtime}.')
        # 保存分解矩阵
        matrix_save(w, h)

    # 数据后处理
    data_postreat()

    return


if __name__ == "__main__":
    main()
