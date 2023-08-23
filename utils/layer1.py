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
        #print("seq_fts1:",seq_fts.shape)#seq_fts1: (1, 10242, 16)

        # simplest self-attention possible
        f_1 = conv1d(seq_fts, 1, 1)
        #print("f_1A:",f_1.shape)#f_1A: (1, 10242, 1)
        f_2 = conv1d(seq_fts, 1, 1)
        #print("f_2A:", f_2.shape)#f_1A: (1, 10242, 1)
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        #print("f_1B:", f_1.shape)#f_2B: ( 10242, 1)
        f_2 = tf.reshape(f_2, (nb_nodes, 1))
        #print("f_2B:", f_2.shape)#f_2B: (10242, 1)

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])


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
        #print("seq_fts2:", seq_fts.shape)#(10242, 16)      (10242,36)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        #print("ret:",ret.shape)#ret: (1, 10242, 16)   (1, 10242, 36)
        #print("seq:", seq.shape)#seq: (1, 10242, 6)    (1, 10242, 512)
        #print("seq_fts3:", seq_fts.shape)#seq_fts: (10242, 16)   (10242, 36)


        if residual:
            if seq.shape[-1] != ret.shape[-1]:

                seq = tf.expand_dims(seq, axis=0)
                #seq = tf.placeholder(tf.float32, shape=[None, seq.shape[0], seq.shape[1], seq.shape[2]])
                # 将(1,10242,6)扩充为4维(None,1,10242,6)
                print("seq1:",seq.shape)
                #扩充通道数目?需要扩充么?理由是啥子?
                seq =tf.layers.conv2d(seq,filters=128, kernel_size=3, padding='SAME')  # 先做一层卷积来增加通道数
                #(?,1,10242,6) 变成 (?,1,10242,128)
                ###将seq(seq_in)作为输入，输出维度是seq的通道数,得到得seq_output进行相加得操作
                seq = Squeeze_excitation_layer(1, seq, int(seq.shape[-1]), 4, layer_name='squeeze_layer_1_0')
                print("seq2:", seq.shape)
                print("iiiiiiiiiiiii-------")

                #(?,1,10242,128) 移除? 变成 (1,10242,128)
                #seq = tf.placeholder(tf.float32, shape=[seq.shape[1], seq.shape[2], seq.shape[3]])
                seq =  tf.squeeze(input=seq, axis=[0])
                print("seq3:", seq.shape)
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
                print("ret:", ret.shape)#(1,10242,36)

            else:
                print("oooooooooooooooooooooooookkkkkkkkkkkkkkkkkkkkkkkk")
                ret = ret + seq

        ret=tf.layers.batch_normalization(ret,axis=1,momentum=0.99,epsilon=0.001,training=training)



        return activation(ret)














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

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units=36, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    squeeze = Global_Average_Pooling(input_x)
    excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
    excitation = Relu(excitation)
    excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
    excitation = Sigmoid(excitation)
    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation
    return scale



