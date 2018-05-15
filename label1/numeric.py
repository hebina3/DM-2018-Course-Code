import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
float_formatter = lambda x: "%.3f" % x  # 创建float格式的数据
np.set_printoptions(formatter={'float_kind': float_formatter})
data = np.loadtxt("magic04.txt", delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)) # 从.txt格式的文件中获取数据并创建矩阵
print("原始数据------------------------\n",data)
col_number = np.size(data, axis=1)  # 在原始数据矩阵中找列号
row_number = np.size(data, axis=0)  # 在原始数据矩阵中找行号
mean_vector = np.mean(data, axis=0).reshape(col_number, 1)  # 计算均值向量
print("均值向量--------------------\n", mean_vector, "\n")
t_mean_vector = np.transpose(mean_vector)
centered_data_matrix = data - (1 * t_mean_vector)  # 计算中心矩阵
print("中心化矩阵-------------------\n", centered_data_matrix, "\n")
t_centered_data_matrix = np.transpose(centered_data_matrix) # 计算中心矩阵的转置
covariance_matrix_inner = (1 / row_number) * np.dot(t_centered_data_matrix, centered_data_matrix)#在打印函数中以下面的描述作为字符串
print(
    "样本协方差矩阵作为中心数据矩阵列之间的内积----------------------------------------\n",
    covariance_matrix_inner, "\n")
def sum_of_centered_points():    # 找到中心数据点的总和
    sum = np.zeros(shape=(col_number, col_number))
    for i in range(0, row_number):
        sum += np.dot(np.reshape(t_centered_data_matrix[:, i], (-1, 1)),
                      np.reshape(centered_data_matrix[i, :], (-1, col_number)))
    return sum
covariance_matrix_outer = (1 / row_number) * sum_of_centered_points()
print(    "样本协方差矩阵作为中心数据点之间的外积 ----------------------------------------\n",
    covariance_matrix_outer, "\n")
vector1 = np.array(centered_data_matrix[:, 1])
vector2 = np.array(centered_data_matrix[:, 2])
def unit_vector(vector):    # 计算属性向量的单位向量
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):   # 计算属性向量之间的角度
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
correlation = math.cos(angle_between(vector1, vector2))    # 计算属性之间的相关性
print("属性1和2之间的相关性: %.5f" % correlation, "\n")
variance_vector = np.var(data, axis=0)   # 创建方差向量
max_var = np.max(variance_vector)        # 找到最大差异
min_var = np.min(variance_vector)        # 找到最小差异
for i in range(0, col_number):           # 找到最大方差指数
    if variance_vector[i] == max_var:
        max_var_index = i;
for i in range(0, col_number):           # 找到最小方差指数
    if variance_vector[i] == min_var:
        min_var_index = i
print(" 最大方差 = %.3f ( 属性 %d )" % (max_var, max_var_index))
print(" 最小方差 = %.3f ( 属性 %d )\n" % (min_var, min_var_index))
covariance_matrix = np.cov(data, rowvar=False)      # 计算协方差矩阵
max_cov=np.max(covariance_matrix)       # 找出协方差矩阵中的最大值
min_cov=np.min(covariance_matrix)       # 找出协方差矩阵中的最小值
for i in range(0, col_number):          # 找到最大值和最小值的索引
    for j in range(0, col_number):
        if covariance_matrix[i, j] == max_cov:
            max_cov_atrr1=i
            max_cov_attr2=j
for i in range(0, col_number):          # 找到最大值和最小值的索引
    for j in range(0, col_number):
        if covariance_matrix[i, j] == min_cov:
            min_cov_atrr1 = i
            min_cov_attr2 = j
print("最大协方差 = %.3f (属性之间 %d and %d)" %(max_cov,max_cov_atrr1,max_cov_attr2))      # 找到最大协方差指数
print("最小协方差 = %.3f (属性之间 %d and %d)\n" %(min_cov,min_cov_atrr1,min_cov_attr2))      # 找到最小协方差指数
df = pd.DataFrame(data[:, 1])   # 为绘图创建数据框
plt.title('Probability Density Function')
plt.xlabel('v1')
plt.ylabel('Probability')
plt.show(plt.scatter(data[:, 1], data[:, 2], c=("lightblue", "pink")))       # 绘制属性之间的散点图
plt.show(df.plot(kind='density'))                # 绘制概率密度函数

