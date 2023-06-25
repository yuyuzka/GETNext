import pickle

import pandas as pd
import time as t
import networkx as nx
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
import numpy as np
#test (10374,15) train(83228,15) #val(10339)

from sklearn.neighbors import NearestNeighbors


class Dataset():
    def __init__(self):
        self.catnameMap = {}
        self.data = []
        self.pois = []
        self.poiId2count = {}
        self.userseqs = {}

    def processDataset(self,origindata):
        # df = pd.read_csv(origindata)
        pois = []
        users2id = {}
        pois2id = {}
        cats2id = {}
        id2pois = {}
        id2users={}
        for checkin in open(origindata+".csv",encoding='utf-8'):
            checkins = checkin.split('\t')
            if (len(checkins) != 8):
                continue
            userID, poiId, catId, catName, lat, lon, time_off, time = checkins
            if poiId not in pois2id:
                pois2id[poiId] = len(pois2id)
                id2pois[len(pois2id)] = poiId
                self.poiId2count[poiId] = 1
            else:
                self.poiId2count[poiId] += 1
        pois2id.clear()
        for checkin in open(origindata + ".csv", encoding='utf-8'):
            checkins = checkin.split('\t')
            if (len(checkins) != 8):
                continue
            userID, poiId, catId, catName, lat, lon, time_off, time = checkins
            count = self.poiId2count[poiId]
            if self.poiId2count[poiId] <10:
                continue
            if userID not in users2id:
                users2id[userID] = len(users2id)
                id2users[users2id[userID]] = userID
                self.userseqs[users2id[userID]] = list()
            if catId not in cats2id:
                cats2id[catId] = len(cats2id)
                self.catnameMap[cats2id[catId]] = catName
            lat = float(lat)
            lon = float(lon)
            if poiId not in pois2id:
                pois2id[poiId] = len(pois2id)
                id2pois[len(pois2id)] = poiId
                self.poiId2count[pois2id[poiId]] = 1
                self.pois.append((pois2id[poiId], lat, lon, cats2id[catId], self.poiId2count[poiId]))
            time = t.strptime(time, '%a %b %d %H:%M:%S +0000 %Y\n')
            self.data.append((users2id[userID], pois2id[poiId], time))
            self.userseqs[users2id[userID]].append((pois2id[poiId],time))

        users2id.clear()
        tempUserseq = {}
        for userseq in self.userseqs:
            if(len(self.userseqs[userseq]) >= 10):
                users2id[id2users[userseq]]=len(tempUserseq)
                tempUserseq[len(tempUserseq)] = self.userseqs[userseq]


        self.userseqs = tempUserseq

        print("pois_count", len(self.pois))
        print("user_count",len(self.userseqs))
        # 保存原始poiId转换为顺序ID的字典
        with open(origindata+"_pois2id.data","wb") as f:
            pickle.dump(pois2id,f)
        # 保存原始userId转换为顺序ID的字典
        with open(origindata+"_users2id.data","wb") as f:
            pickle.dump(users2id,f)
        #保存原始catID转换为顺序Id的字典
        with open(origindata+"_cats2id.data","wb") as f:
            pickle.dump(cats2id,f)
        # 保存经过冷启动过滤后的POI(顺序后的POIid,lat,lon,顺序后的caiId,poi的访问频率）
        with open(origindata+"_pois.data","wb") as f:
            pickle.dump(self.pois,f)
        #保存经过处理后的签到记录（userId,POIID,time)
        with open(origindata+"_checkins.data","wb") as f:
            pickle.dump(self.data,f)
        #保存经过处理后的用户访问序列（从0开始，(poiID,time)
        with open(origindata + "_userseq.data", "wb") as f:
            pickle.dump(self.userseqs, f)

# 仅构建连接关系
# 根据访问轨迹构建图的邻接关系
def build_next_adj(origin_file):
    userSeqs = pd.read_pickle(origin_file+"_userseq.data")
    G = nx.DiGraph()
    loop = tqdm(userSeqs)
    for user_id in loop:
        user_seq = userSeqs[user_id]
        # Add edges (Check-in seq)
        previous_poi_id = -1
        for poi_id, time in user_seq:

            # No edge for the begin of the seq or different traj
            if previous_poi_id == -1:
                previous_poi_id = poi_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                continue
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id)
            previous_poi_id = poi_id
    print(G.edges())
    with open(origin_file + "_adjNext.data", "wb") as f:
        pickle.dump(G.edges(), f)
    return G

def haversine(x, y):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1=x[1]
    lon1 = x[2]
    lat2=y[1]
    lon2=y[2]
    # 将十进制度数转化为弧度
    lat1, lon1, lat2,lon2, = map(radians, [ lat1, lon1, lat2,lon2])
    # haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r

#读取含有顺序访问关系的POI邻接数据，并建立K近邻关系的邻接数据
def build_dist_adj(origin_file,k):
    pois = pd.read_pickle(origin_file + "_pois.data")
    next_adj = pd.read_pickle(origin_file+"_adjNext.data")
    G = nx.DiGraph()
    G.add_edges_from(next_adj)
    #计算距离
    nbrs = NearestNeighbors(n_neighbors=k, metric=haversine).fit(pois)
    distances, indices = nbrs.kneighbors(pois)
    num_nodes = len(pois)

    # 构建邻接矩阵
    for i in range(num_nodes):
        neighbors = indices[i]
        for neighbor in neighbors:
            if G.has_edge(i,neighbor):
                continue
            else:
                G.add_edge(i,neighbor)
    with open(origin_file + "_adj_K="+k+".data_", "wb") as f:
        pickle.dump(G.edges(), f)
    return G

#根据距离构建图的邻接关系

if __name__ == '__main__':

    origin_file = "./dataset/NYC/dataset_NYC"
    dataset = Dataset()
    dataset.processDataset(origin_file)
    # build_next_adj(origin_file)
    # build_dist_adj(origin_file,30)

