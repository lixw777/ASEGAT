import tensorflow as tf
conv1d = tf.layers.conv1d
#import utils.SE_layer as SE

def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop, coef_drop, residual=False, training=True):
    ###由于training是bool类型的，所以需要给它赋值，后面会对其进行覆盖的
    #print("training_test:",training,type(training))
    # out_sz=hid_units[0] 从gat.py/sp_gat.py中得到的
    # seq 指的是输入的节点特征矩阵，大小为 [num_graph, num_node, fea_size]  seq_in: (1, 10242, 6)
    # out_sz 指的是变换后输出的节点特征维度，也就是Whi后的节点表示维度，F。   16
    # bias_mat 是经过变换后的邻接矩阵（N,N）的掩码矩阵，大小为 [num_graph,num_node, num_node]
    # in_drop: 输入的dropout率
    # coef_drop: 注意力矩阵的dropout率

    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        # 最简单的自我注意可能
        f_1 = conv1d(seq_fts, 1, 1)
        f_2 = conv1d(seq_fts, 1, 1)
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        print("f_1C:",f_1.shape)#f_1A: (1, 10242, 1)
        f_2 = adj_mat * tf.transpose(f_2, [1,0])
        print("f_2C:", f_2.shape)  # f_1A: (1, 10242, 1)


        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)

        coefs = tf.sparse_softmax(lrelu)
        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])

        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
               ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        ret=tf.layers.batch_normalization(ret,axis=1,momentum=0.99,epsilon=0.001,training=training)
        return activation(ret)

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,training=True):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        logits = tf.nn.leaky_relu(logits)

        print("logits:",logits.shape,type(logits),logits.dtype)
        #logits: (1, 10242, 10242) <class 'tensorflow.python.framework.ops.Tensor'>  <dtype: 'float32'>
        #logits：一个非空的Tensor。必须是下列类型之一：half， float32，float64
        print("bias_mat:",bias_mat,type(bias_mat),bias_mat.dtype)
        coefs = tf.nn.softmax( logits+ tf.sparse_to_dense(bias_mat,[10242,10242],1.0, 0.0))

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        ret = tf.layers.batch_normalization(ret, axis=1, momentum=0.99, epsilon=0.001, training=training)

        return activation(ret)  # activation












import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))





