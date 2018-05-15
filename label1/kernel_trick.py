import numpy as np
import csv
import pandas as pd
#读取数据文件
data=pd.read_csv('iris.csv')
data=np.array(data)
data=np.mat(data[:,0:4])
length=len(data)
#使用核函数计算内核（k）
k=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k[i,j]=(np.dot(data[i],data[j].T))**2
        k[j,i]=k[i,j]
#保存核矩阵
name=range(length)
test=pd.DataFrame(columns=name,data=k)
test.to_csv('iris_k.csv')
len_k=len(k)
#中心化核矩阵
I=np.eye(len_k)
one=np.ones((len_k,len_k))
A=I-1.0/len_k*one
centered_k=np.dot(np.dot(A,k),A)
print(centered_k)
#保存中心化核矩阵
test=pd.DataFrame(columns=name,data=centered_k)
test.to_csv('iris_ck.csv')
#归一化核矩阵
W_2=np.zeros((len_k,len_k))
for i in range(0,len_k):
    W_2[i,i]=k[i,i]**(-0.5)
normalized_k=np.dot(np.dot(W_2,k),W_2)
# 保存归一化的核心矩阵
test=pd.DataFrame(columns=name,data=normalized_k)
test.to_csv('iris_nk.csv')
#计算φ
fai=np.mat(np.zeros((length,10)))
for i in range(0,length):
    for j in range(0,4):
        fai[i,j]=data[i,j]**2
    for m in range(0,3):
        for n in range(m+1,4):
            j=j+1
            fai[i,j]=2**0.5*data[i,m]*data[i,n]
#保存φ
name_f=range(10)
test=pd.DataFrame(columns=name_f,data=fai)
test.to_csv('iris_fai.csv')
#通过φ计算内核
k_f=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_f[i,j]=(np.dot(fai[i],fai[j].T))
        k_f[j,i]=k_f[i,j]
test=pd.DataFrame(columns=name,data=k_f)
test.to_csv('iris_kf.csv')
#中心化φ
rows=fai.shape[0]
cols=fai.shape[1]
centered_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,cols):
    centered_fai[:,i]=fai[:,i]-np.mean(fai[:,i])
print(centered_fai)
test=pd.DataFrame(columns=name_f,data=centered_fai)
test.to_csv('iris_cf.csv')
#通过中心化φ计算中心化核矩阵
k_cf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_cf[i,j]=(np.dot(centered_fai[i],centered_fai[j].T))
        k_cf[j,i]=k_cf[i,j]
test=pd.DataFrame(columns=name,data=k_cf)
test.to_csv('iris_kcf.csv')
#归一化φ
normalized_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,len(fai)):
    temp=np.linalg.norm(fai[i])
    normalized_fai[i]=fai[i]/np.linalg.norm(fai[i])
print(normalized_fai)
test=pd.DataFrame(columns=name_f,data=normalized_fai)
test.to_csv('iris_nf.csv')
#通过归一化φ计算归一化核矩阵
k_nf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_nf[i,j]=(np.dot(normalized_fai[i],normalized_fai[j].T))
        k_nf[j,i]=k_nf[i,j]
test=pd.DataFrame(columns=name,data=k_nf)
test.to_csv('iris_knf.csv')
