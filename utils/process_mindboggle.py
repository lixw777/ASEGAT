import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
from scipy.sparse import lil_matrix
import random
import tensorflow as tf
from sklearn.model_selection import KFold
import glob
import pandas as pd

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""



def generate_necessary_data_sparse(label_path, feat_path):
    '''return one subject's adj,features,labels'''
    regex = re.compile(r'index=([0-9]*?),')

    features = pd.read_csv(feat_path, encoding='utf-8', header=None, sep=' ',
                           index_col=0)
    assert not features.isnull().sum().sum()

    features = lil_matrix(features.values)
    features_index = features.astype('float32')

    data = np.loadtxt(label_path).astype(int)
    B = np.zeros((10242,36))
    for i in range(10242):
        B[int(data[i, 0]), int(data[i, 1])] = 1
    labels_index = B.astype('int32')

    '''对特征进行归一化处理，或者直接不对其进行一个归一化处理'''
    #print("features_index:",features_index.shape,type(features_index),features_index.dtype)    #features_index: (10242, 12) <class 'scipy.sparse.lil.lil_matrix'> float32
    features_index, spars = preprocess_row_features(features_index)
    '''对前11个特征进行行归一化，对最后1进行列归一化
       若是针对12个特征不做初始化，就会导致验证的数据准确率很低'''
    #features_index1,spars=preprocess_row_features(features_index[:,[0,1,2,3,4,5,6,7,8,9,10]])
    #features_index2, spars = preprocess_col_features(features_index[:,[11]])
    #features_index=np.column_stack((features_index1,features_index2))
    #features_index=features_index.todense()


    return labels_index, features_index


def get_num_subjects(pathname):
    path_file_number = glob.glob(pathname)  # 获取当前文件夹下个数   #print(path_file_number)
    num_subject = len(path_file_number)
    print("num_subject:", num_subject)
    return num_subject


def split_data_sparse_Kfold(num_subject,features_data, labels_data, nb_nodes):
    '''return y_train(num_subject,nb_node,nb_classes);y_test(num_subject,nb_node,nb_classes)
              train_mask(num_subject,) or (1,numsubject)
              split the num_subject not the nb_node'''
    # 将其转为list
    subject_idx = []
    for index in range(1, num_subject + 1):
        subject_idx = subject_idx + [index]

    data = subject_idx
    data = np.array(data)
    print("data:",data,type(data))

    kf = KFold(n_splits=num_subject, shuffle=True)

    for train_list, test_list in kf.split(data):
        train_list = train_list.tolist()
        test_list = test_list.tolist()
        print("train_list:", train_list)
        print("test_list:", test_list)
        x_train = features_data[train_list, :, :]
        x_test = features_data[test_list, :, :]
        y_train = labels_data[train_list, :, :]
        y_test = labels_data[test_list, :, :]

        return y_train, y_test, x_train, x_test,test_list




def generate_3D_data_sparse(num_subject):
    '''return num_subject's 3D data :
    adj(num_subject,nb_node,node)  features(num_subject,nb_node,feature_size) labels(num_subject,nb_node,nb_classes)'''
    # 构建一个全零的矩阵,用于将79个二维的adj,feature,label装进去，形成一个三维的
    features_data = np.empty((num_subject, 10242, 6),dtype=float)
    labels_data = np.empty((num_subject, 10242, 36),dtype=int)

    for i in range(num_subject):
        index = str(i + 1)
        label_path = 'data/subject' + index + '/lhlabels.txt'
        feat_path = 'data/subject' + index + '/lhsixfeatures.txt'
        labels_index, features_index = generate_necessary_data_sparse( label_path, feat_path)
        features_data[i, :, :] = features_index
        labels_data[i, :, :] = labels_index

    return features_data, labels_data

def generate_adj(edge_path):
    '''return one subject's adj,features,labels'''
    edges = pd.read_csv(edge_path, encoding='utf-8', header=None,
                        sep=' ', names=['source', 'target'])
    assert not edges.isnull().sum().sum()
    graph = nx.Graph()
    edge_tuple = [(row.source, row.target) for row in edges.itertuples(index=False)]
    graph.add_edges_from(edge_tuple)
    adj_index = nx.adjacency_matrix(graph)
    indices, adj_data, adj_shape = preprocess_adj_bias(adj_index)

    return adj_data



###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_row_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))#print(rowsum,rowsum.shape)#(10242, 1)
    r_inv = np.power(rowsum, -1).flatten()  # a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降# (10242,)
    #print("r:",r_inv.shape,type(r_inv),r_inv.dtype)#(10242,) <class 'numpy.ndarray'> float32
    r_inv[np.isinf(r_inv)] = 0.#(10242,)
    r_mat_inv = sp.diags(r_inv)   #(10242, 10242)
    features = r_mat_inv.dot(features)#10242 10242* 10242 11=(10242, 11)
    return features.todense(), sparse_to_tuple(features)

def preprocess_col_features(features):
    col=features.todense()##(10242, 1) <class 'numpy.matrix'> float32
    colsum = np.array(features.sum(0))#每列相加[[5330.]]
    r_inv = np.power(colsum, -1)#.flatten(('F'))#a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降#(10242,)
    r_inv = np.multiply(col,r_inv)#(10242, 1)--->(10242,)
    print("r:", r_inv.shape, type(r_inv), r_inv.dtype)#(10242, 1) <class 'numpy.matrix'> float32
    #r_inv[np.isinf(r_inv)] = 0.#r2: (1,)
    #r_inv[np.isnan(r_inv)] = 0.#(10242,)
    r_mat_inv = sp.diags(r_inv)#(10242,10242)
    features = r_mat_inv.dot(features)#10242 10242* 10242 1=(10242, 11)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


def all_node_mask(l):
    mask = np.ones(l)
    mask = mask[np.newaxis]
    return (np.array(mask, dtype=np.bool))


def split_data_sparse(num_subject,features_data, labels_data, nb_nodes):
    '''return y_train(num_subject,nb_node,nb_classes);
              y_test(num_subject,nb_node,nb_classes)
              train_mask(num_subject,) or (1,numsubject)
              split the num_subject not the nb_node'''
    # 将其转为list
    subject_idx = []
    #num_subject=90
    for index in range(num_subject-10):
        subject_idx = subject_idx + [index]

    id_list = random.sample(subject_idx, len(subject_idx))  # reorder

    t1 = int((8/9)* len(id_list))
    train_list = id_list[:t1]
    val_list = id_list[t1:]
    test_list=[90,91,92,93,94,95,96,97,98,99]

    print("train_list:", train_list,len(train_list))
    print("val_list:",val_list,len(val_list))
    print("test_list:", test_list, len(test_list))
    '''
    train_list: [93, 43, 12, 19, 33, 1, 87, 32, 4, 68, 89, 30, 6, 2, 65, 72, 66,
                 35, 58, 42, 52, 94, 64, 25, 3, 76, 28, 39, 69, 9, 55, 81, 80, 67, 
                 82, 46, 27, 75, 14, 98, 62, 10, 5, 26, 56, 96, 48, 99, 77, 57, 37, 
                 54, 61, 53, 22, 11, 85, 90, 63, 34, 50, 73, 51, 86, 59, 24, 92, 8, 
                 74, 7, 36, 78, 17, 31, 49, 47, 83, 0, 95, 23] 80
    val_list: [79, 40, 91, 45, 88, 84, 18, 60, 13, 71] 10
    test_list: [16, 21, 41, 15, 20, 97, 38, 29, 70, 44] 10
    '''
    x_train = features_data[train_list, :, :]
    x_val=features_data[val_list,:,:]
    x_test = features_data[test_list, :, :]
    y_train = labels_data[train_list, :, :]
    y_val=labels_data[val_list,:,:]
    y_test = labels_data[test_list, :, :]

    return x_train,x_val, x_test, y_val,y_train, y_test,test_list




