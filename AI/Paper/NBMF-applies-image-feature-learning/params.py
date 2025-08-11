"""
NBMF Applies Image Feature Learning

这个文件用于保存参数。

作者：周澍锦
联系方式：zhoushujin@tongji.edu.cn
"""

# 文件数据参数
file_list_path = 'face/'
result_list_path = 'result/'
magic_number = 'P5'
width = 19
height = 19
max_gray_value = 255

# 数据处理参数
v_select = list(range(0, 60))
columns = [0, 1]

# 退火NSP矩阵分解参数
anneal_nsp_k = 35
anneal_nsp_tol = 1e-6
anneal_nsp_time_limit = 10000
anneal_nsp_max_iter = 100

# 退火BCD矩阵分解参数
anneal_bcd_k = 35
anneal_bcd_max_iter = 10
anneal_bcd_alpha = 0.1

# Gurobi_BCD矩阵分解参数
gurobi_bcd_k = 35
gurobi_bcd_max_iter = 10
gurobi_bcd_alpha = 0.1
gurobi_bcd_time_limit = 1

# CIM真机BCD矩阵分解
save_dir_path = 'tmp/'
machine_name = 'CPQC-550'
cim_bcd_bit = 550
cim_bcd_k = 35
cim_bcd_max_iter = 1
cim_bcd_alpha = 0.1
