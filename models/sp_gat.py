import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class SpGAT(BaseGAttN):
    #定义神经网络的前向传播过程
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,bias_mat, hid_units, n_heads, activation=tf.nn.elu,
            residual=False):
        ''' 第一层，有H1个注意力头，每个头的输入都是(B,N,D)，每头注意力输出(B,N,F1)；将所有注意力头的输出聚合，聚合为(B,N,F1*H1)'''
        attns = []
        # n_heads[0]=第一层注意力的头数，设为H1=32
        """hid_units = [16]  # numbers of hidden units per each attention head in each layer
           n_heads = [32,1]  # additional entry for the output layer"""
        for _ in range(n_heads[0]):
            attns.append(layers.sp_attn_head(inputs,adj_mat=bias_mat,out_sz=hid_units[0], activation=activation,
                                             nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop,
                                             residual=True, training=training))
            #attn是(1,10242,)但是返回有8个头所以是(1,10242,8*8)   [(B,N,F1),(B,N,F1)..]=>(B,N,F1*H1)

        h_1 = tf.concat(attns, axis=-1)
        #print("h_1:",h_1.shape)#h_1: (1, 10242, 512)

        h_1 = tf.expand_dims(h_1, axis=0)
        h_1 = Squeeze_excitation_layer(h_1, int(h_1.shape[-1]), 8, layer_name='RC')       
        h_1 = tf.squeeze(input=h_1, axis=[0])


        '''中间层，层数是 len(hid_units)-1；第i层有Hi个注意力头，输入是 (B,N,F1*H1)，每头注意力输出是 (B,N,Fi)；每层均聚合所有头注意力，得到 (B,N,Fi*Hi)'''
        # len(hid_units)=中间层的个数32
        for i in range(1, len(hid_units)):
            print("lllllllllllllllllllllllllllllll")
            attns = []
            # n_heads[i]=中间第i层的注意力头数，设为Hi
            for _ in range(n_heads[i]):
                attns.append(layers.sp_attn_head(h_1,
                    adj_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=True,training=training))
            # [(B,N,Fi),(B,N,Fi)..]=>(B,N,Fi*Hi)
            h_1 = tf.concat(attns, axis=-1)
        #print("h_3:",h_1.shape)#h_3: (1, 10242, 1024)

        '''最后一层，共 n_heads[-1] 头注意力，一般为1；输入：最后一个中间层的输出(B,N,Fi*Hi)输出：(B,N,C)，C是分类任务中的类别数'''
        out = []
        #print("h_4", h_1.shape)#h_4 (1, 10242, 1024)
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(h_1, adj_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=True,training=training))

        # 将多头注意力的结果相加，并取平均
        logits = tf.add_n(out) / n_heads[-1]
        #print("logits:",logits.shape,type(logits))  #logits: (1, 10242, 36) <class 'tensorflow.python.framework.ops.Tensor'>

        return logits



import tensorflow as tf
from   tflearn.layers.conv import global_avg_pool

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Squeeze_excitation_layer( input_x, out_dim, ratio, layer_name):
    squeeze = Global_Average_Pooling(input_x)
    excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'fully_connected1')
    excitation = Relu(excitation)
    excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'fully_connected2')
    excitation = Sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation
    return scale
