"""
NBMF Applies Image Feature Learning

这个文件用于数据预处理。

作者：周澍锦
联系方式：zhoushujin@tongji.edu.cn
"""

import os
import numpy as np
import params

file_list_path = params.file_list_path


def pgm_read_binary(pgm_path):
    """
    读取二进制pgm图像文件

    Args:
        pgm_path: string, pgm文件路径

    Returns:
        pixels: int * int, pgm文件像素向量
    """
    # 根据文件编码方式（P5二进制）打开
    with open(pgm_path, 'rb') as f:
        # 读取头部信息，包括魔数、尺寸、最大灰度值等
        magic_number = f.readline().decode().strip()
        width, height = map(int, f.readline().decode().strip().split())
        max_gray_value = int(f.readline().decode().strip())

        # 读取像素数据
        pixels = list(f.read())

        f.close()

    return pixels


def data_pretreat():
    """
    数据预处理，以矩阵形式保存

    Args:

    Returns:
        v: int * int, 整体数据矩阵
    """
    global file_list_path
    pixels_matrix = []
    # 替换为实际数据路径
    file_list = os.listdir(file_list_path)
    # 导入并集成数据
    for file_name in file_list:
        file_path = file_list_path + file_name
        pixels = pgm_read_binary(file_path)
        pixels_matrix.append(pixels)

    # 将像素列表转换为NumPy数组
    v = np.array(pixels_matrix).T
    # 将矩阵保存为二进制文件
    np.save('matrix_v.npy', v)
    # 将矩阵保存为表单文件，设置分隔符和数据类型
    np.savetxt('matrix_v.csv', v, delimiter=',', fmt='%d')

    return v

