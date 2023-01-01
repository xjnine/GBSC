# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:36:29 2022

@author: kwy
"""
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D, axes3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.cluster import k_means
import numpy.linalg as la
import datetime





class GB:
    def __init__(self, data, label):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.center = self.data.mean(0)# According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        self.radius = self.get_radius()
        self.flag = 0
        self.label = label
        self.num = len(data)
        self.out = 0
        self.size = 1
        self.overlap = 0
        self.hardlapcount = 0 
        self.softlapcount = 0
    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)




def get_radius_2(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    mean_radius = 0
    radius = max(distances)
    for i in distances:
        mean_radius = mean_radius + i
    mean_radius = mean_radius/num
    return radius
    

def get_separation(gb1,gb2):
    min_dis = 1
    for i in range(0,len(gb1)):
        for j in range(0,len(gb2)):
            temp_dis =  ((gb1[i] - gb2[j]) ** 2).sum(axis=0) ** 0.5
            if min_dis > temp_dis:
                min_dis = temp_dis
    return min_dis

def division_2_2(gb_list,n):
    gb_list_new_2 = []
    for gb in gb_list:
        if(len(gb) >=8):
           ball_1,ball_2 = spilt_ball_2(gb)
           density_parent = get_density_volume(gb)
           density_child_1 = get_density_volume(ball_1)
           density_child_2 = get_density_volume(ball_2)
           w = len(ball_1)+len(ball_2)
           w1 = len(ball_1)/w
           w2 = len(ball_2)/w
           w_child = (w1*density_child_1+w2*density_child_2)
           t1 = ((density_child_1 > density_parent)&(density_child_2 > density_parent))
           t2 = (w_child > density_parent)
           t3 = ((len(ball_1) >4) & (len(ball_2) >4))
           if (t2):
               gb_list_new_2.extend([ball_1,ball_2])
           else:
               gb_list_new_2.append(gb)
        else:
          gb_list_new_2.append(gb)  
        
    return gb_list_new_2

def judge_contain(gb_target,gb_list):
    gb_list_new = []
    center_target = gb_target.mean(0)
    radius_target = get_radius(gb_target)
    for gb in gb_list:
        if(gb is gb_target):
            continue
        else:
            center_gb = gb.mean(0)
            radius_gb = get_radius(gb)
            distance_1 =  np.linalg.norm(center_target-center_gb)
            if(radius_target >= (radius_gb + distance_1)):
                return 1
    return 0

def move_contain(gb_list):
    gb_list_final=[]
    for gb in gb_list:
        if 1:
            is_contain = judge_contain(gb,gb_list)
            if (is_contain):
              gb_list_final.extend(spilt_ball_2(gb))
              print("有包含关系",len(gb_list_final))
            else:
              gb_list_final.append(gb)
        else:
            gb_list_final.append(gb)
              
    return gb_list_final

def spilt_ball(data):
    cluster = k_means(X=data,init='k-means++', n_clusters=2)[1]
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]

def spilt_ball_2(data):
    ball1 = []
    ball2 = []
    n,m = data.shape
    X = data.T
    G = np.dot(X.T,X)
    H = np.tile(np.diag(G),(n,1))
    D = np.sqrt(H + H.T-G*2)
    r,c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(0,len(data)):
        if D[j,r1] < D[j,c1]:
            ball1.extend([data[j,:]])
        else:
            ball2.extend([data[j,:]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_density_volume(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius/num
    dimension = len(gb[0])
    if mean_radius!=0:
        density_volume = num/(sum_radius)
    else:
        density_volume = num

    return density_volume

def get_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius
    

def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:,0], data[:,1],s = 7, c = "#314300", linewidths=5, alpha=0.6, marker='o',label='data point')
    plt.legend()

def gb_plot(gbs,noise):
    color = {
        0: '#fb8e94',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#df7a30',
        15: '#d9d9fc',
        16: '#c490a0',
        17: '#cf8878',
        18: '#e26538',
        19: '#f28a63',
        20: '#f2debd',
        21: '#e96d29',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444',}
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25'}

    plt.figure(figsize=(10,10))
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label,0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)
    label = set()
    for key in label_num.keys():
        label.add(key)
    list = []
    for i in range(0,len(label)):
        list.append(label.pop())
    for i in range(0,len(list)):
        if list[i] == -1:
           list.remove(-1)
           break
    for i in range(0,len(list)):
        for key in gbs.keys():
            if(gbs[key].label == list[i]):
                plt.scatter(gbs[key].data[:,0], gbs[key].data[:,1],s = 6, c = color[i], linewidths=4.5, alpha=1, marker='o',label=label_c[i])
                break
    for key in gbs.keys():
        for i in range(0,len(list)):
            if(gbs[key].label == list[i]):
                plt.scatter(gbs[key].data[:,0], gbs[key].data[:,1],s = 6, c = color[i], linewidths=4.5, alpha=1, marker='o')
    if (len(noise)>0):
        plt.scatter(noise[:,0],noise[:,1],s =40, c = 'black', linewidths=2, alpha=1, marker='x',label='noise')
    for key in gbs.keys():
        for i in range(0,len(list)):
            if(gbs[key].label == -1):
                plt.scatter(gbs[key].data[:,0], gbs[key].data[:,1],s = 1, c = 'black', linewidths=4, alpha=1, marker='x')

    plt.legend(loc=1,fontsize=12)
    plt.show()

def draw_ball(gb_list):
    for data in gb_list:
        if len(data) >1 :
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-',color='black', lw=0.7)
            
        else:
             plt.plot(data[0][0], data[0][1],marker = '*',color= '#0000EF', markersize=3)
    plt.plot(x, y, ls='-',color='black', lw=1.2,label='granular-ball boundary')
    plt.legend(loc=1)
    plt.show()

def normalized_ball(gb_list,radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if (len(gb)<2):
            gb_list_temp.append(gb)
        else:
            ball_1,ball_2 = spilt_ball_2(gb)
            if(get_radius(gb) <= 2*radius_detect):
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1,ball_2])
    
    return gb_list_temp

def normalized_ball_2(gb_dist,radius_mean,list1):
    gb_list_temp = []
    for i in range(0,len(radius_mean)):
        for key in gb_dist.keys():
            if gb_dist[key].label==list1[i]:
                if (gb_dist[key].num < 2):
                    gb_list_temp.append(gb_dist[key].data)
                else:
                    ball_1,ball_2 = spilt_ball_2(gb_dist[key].data)
                    if(gb_dist[key].radius <= 2*radius_mean[i]):
                        gb_list_temp.append(gb_dist[key].data)
                    else:
                        gb_list_temp.extend([ball_1,ball_2])
    return gb_list_temp



def detect_noise(gb_list):
    noise = []
    gb_list_final = []
    for gb in gb_list:
        if (len(gb) <=1):
            noise.extend(gb)
        else:
            gb_list_final.append(gb)
    return [gb_list_final,noise]


def getGranularBall(key):

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.color_palette("bright", 10)


    dict_mat = loadmat(r"F:\python\project\hyper\GB\dataset\\data_" + key + ".mat")  # 加载数据集

    data = dict_mat['fea']
    target = dict_mat['gt']
    target = target.T
    target = target[0]

    data_num = data.shape[0]
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

    index = np.full((1,data.shape[0]),0)
    data_index = np.insert(data,data.shape[1],values=index,axis=1)
    gb_list_temp = [data]
    row = np.shape(gb_list_temp)[0]
    col = np.shape(gb_list_temp)[1]
    n = row*col


    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_2_2(gb_list_temp,n) #质量轮
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []
    for gb in gb_list_temp:
        if(len(gb)>=2):
            radius.append(get_radius(gb))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median,radius_mean)
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list_final = gb_list_temp
    return target,data,gb_list_final,data_num

    

