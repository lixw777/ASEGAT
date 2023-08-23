import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN
'''先看gat.py的框架，再慢慢看其他的代码'''

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):

        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False,training=True))# return activation(ret)

        h_1 = tf.concat(attns, axis=-1)#将其拼接起来

        ##加上输出层
        out = []
        for i in range(n_heads[-1]):
            #n_heads[-1]=1最后一个元素
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False,training=True))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits
