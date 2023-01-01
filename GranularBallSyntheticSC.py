import gc
import os
import psutil
import time
from math import exp
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from GB.GranularBallSynthetic import getGranularBall


class GB:
    def __init__(self, data, label):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.center = self.data.mean(0)# According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        self.label = label
        self.radius = self.get_radius()
    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


class Da:
    def __init__(self, data, label):
        self.data = data
        self.label = label


def division(gb_list,density_threshold):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >4:
            density = get_density(gb)
            if density > density_threshold:
                gb_list_new.extend(spilt_ball(gb))
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new


def spilt_ball(data):
    cluster = k_means(X=data, n_clusters=2)[1]
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]




def get_radius(self):
    diffMat = np.tile(self.center, (self.num, 1)) - self.data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    return max(distances)


def add_center(gb_list):
    gb_dist = {}
    for i in range(0,len(gb_list)):
        gb = GB(gb_list[i],i)
        gb_dist[i] = gb
    return gb_dist

def gb_plot(gbs):
    color = {
        0: '#0000FF',
        1: '#FFFF00',
        2: '#FF0000',
        3: '#FFC0CB',
        4: '#8B0000',
        5: '#000000',
        6: '#008000',
        7: '#FFD700',
        8: '#A52A2A',
        9: '#FFA500',
        10: '#00FFFF',
        11: '#FF00FF',
        12: '#F0FFFF',
        13: '#00FFFF'}
    plt.figure(figsize=(10,10))
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label,0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)

    label = set()
    for key in label_num.keys():
        if label_num[key]>10:
            label.add(key)
    list = []
    for i in range(0,len(label)):
        list.append(label.pop())
    for key in gbs.keys():
        # print(key)
        for i in range(0,len(list)):
            if(gbs[key].label == list[i]):
                if(i<14):
                    for data in gbs[key].data:
                        plt.plot(data[0], data[1],marker = '*',color= color[i], markersize=3)
                    break
    plt.show()


def get_affinitys(data,center,radius1,radius2):
    return np.linalg.norm(data - center)-radius2-radius1


def gb_plot2(gbs,key):
    color = {
        0: '#fb8e94',  # 孟菲斯配色,
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
        25: '#ff8444', }
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

    plt.figure(figsize=(10, 10))  # 图像宽高
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label, 0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)

    label = set()
    for key in label_num.keys():
        label.add(key)
    list = []
    for i in range(0, len(label)):
        list.append(label.pop())

    for i in range(0, len(list)):
        if list[i] == -1:
            list.remove(-1)
            break

    for i in range(0, len(list)):
        for key in gbs.keys():
            if (gbs[key].label == list[i]):
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=6, c=color[i], linewidths=4.5, alpha=1,
                            marker='o')
                break
    for key in gbs.keys():
        for i in range(0, len(list)):
            if (gbs[key].label == list[i]):
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=6, c=color[i], linewidths=4.5, alpha=1,
                            marker='o')

    for key in gbs.keys():
        for i in range(0, len(list)):
            if (gbs[key].label == -1):
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=1, c='black', linewidths=4, alpha=1, marker='x')
    plt.show()


def main():
    keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ress = [2, 2, 4, 2, 3, 2, 6, 4, 5, 4, 3, 3]
    detas = [0.05,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.1,0.05,0.05,0.1]

    for d in range(len(keys)):
        print(keys[d])
        print(detas[d])
        gc.collect()
        pid = os.getpid()
        p = psutil.Process(pid)
        info_start = p.memory_full_info().uss / 1024
        time1 = time.time()
        data, gb_list, data_num = getGranularBall(keys[d])
        gb_dist = add_center(gb_list)

        X = []
        for i in range (len(gb_dist)):
            X.append(gb_dist[i].center)
        R = []
        for i in range (len(gb_dist)):
            R.append(gb_dist[i].radius)
        affinity = []
        for x in range (len(gb_dist)):
            temp = []
            for y in range(len(gb_dist)):
                if(x==y):
                    temp.append(0)
                    continue
                value = -1*get_affinitys(X[x],X[y],R[x],R[y])
                deta = 2 * (np.square(detas[d]))
                temp.append(exp(value/deta))
            affinity.append(temp)
        affinity = np.asarray(affinity)

        clustering = SpectralClustering(n_clusters=ress[d],
                                        affinity="precomputed",
                                        assign_labels="discretize").fit(affinity)

        for i in range(len(gb_dist)):
            gb_dist[i].label = clustering.labels_[i]

        labels = []
        for i in range(0, len(data)):
            for j in range(0,len(gb_dist)):
                if(data[i] in gb_dist[j].data):
                    labels.append(gb_dist[j].label)
                    break

        time2 = time.time()

        print((time2 - time1))

        info_end = p.memory_full_info().uss / 1024
        print("The program is consuming memory" + str(info_end - info_start) + "KB")

        gb_plot2(gb_dist,keys[d])


if __name__ == '__main__':
    main()




