import tensorflow as tf
import numpy as np
from utils import process_mindboggle
import pandas as pd

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        #print("loss:",loss,type(loss))
        #print("lossL2:",lossL2,type(lossL2),lossL2.dtype)
        '''loss: Tensor("add_51212:0", shape=(), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
           lossL2: Tensor("mul_20489:0", shape=(), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'> <dtype: 'float32'>'''
        train_op = opt.minimize(loss + lossL2)#考虑两个损失加起来的最小值
        #train_op = opt.minimize(tf.add(loss , lossL2))

        return train_op

    def preshape(logits, labels, nb_classes):
        ''' print("preshape_logits:", logits, logits.shape)
        print("preshape_labels:", labels, labels.shape)'''
        #change the logits and labels shape to match the nb_classes
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

##########################
# Adapted from tkipf/gcn #
##########################

    def masked_softmax_cross_entropy(logits, labels, mask):

        """Softmax cross-entropy loss with masking.
        交叉损失logits: Tensor("Reshape:0", shape=(10242, 76), dtype=float32) (10242, 76)
        交叉损失labels: Tensor("Reshape_1:0", shape=(10242, 76), dtype=int32) (10242, 76)
        交叉损失mask: Tensor("Reshape_2:0", shape=(10242,), dtype=int32) (10242,)"""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        loss1=tf.reduce_mean(loss)
        return loss1


    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def create_triangle_adj(triangle, batch_size, labels):
        new_tensor_list = []
        for i in range(triangle.shape[0]):
            m = triangle[i][0]
            n = triangle[i][1]
            m1 = labels[m]  # ()
            n1 = labels[n]
            new_tensor_list.append(m1)
            new_tensor_list.append(n1)

        # new_tensor_list   <class 'list'> 12
        new_tensor = tf.convert_to_tensor(new_tensor_list)  # (12,)
        new_tensor = tf.reshape(new_tensor, [triangle.shape[0], 2])
        new_tensor = tf.cast(new_tensor, dtype=tf.int32)  # (6,2)
        onehot_labels = tf.sparse_to_dense(new_tensor, tf.stack([batch_size, batch_size]), 1.0, 0.0,
                                           validate_indices=False)
        return onehot_labels



    def NoAdjLoss(pred,labels):
        '''两个相邻脑区用来约束分割，
        想法就是构建预测标签和真实标签的邻接矩阵，进而对其使用交叉熵作为损失进行处理'''
        '''labels: (10242, 36)    pred: (10242, 36)'''
        labels = tf.argmax(labels, 1)#(10242,)
        preds = tf.argmax(pred, 1)#(10242,)
        batch_size = tf.reduce_max(labels)+1#最大的数是35，但是这里的矩阵应该是36
        '''indices is out of bounds'''
        array = np.ones((10242))
        array[10241]=0
        A=array
        array = np.ones((10242))
        array[0] = 0
        B=array
        mask1 = (np.array(A, dtype=np.bool))#[ True  True  True ...  True  True False] (10242,) <class 'numpy.ndarray'> bool
        mask2 = (np.array(B, dtype=np.bool))#[False  True  True ...  True  True  True] (10242,) <class 'numpy.ndarray'> bool
        a1 = tf.boolean_mask(preds, mask=mask1)
        b1 = tf.boolean_mask(preds, mask=mask2)
        a2 = tf.boolean_mask(labels, mask=mask1)
        b2 = tf.boolean_mask(labels, mask=mask2)
        preds_indice = tf.expand_dims(a1, 1)#Tensor("ExpandDims:0", shape=(?, 1), dtype=int64) (?, 1) <class 'tensorflow.python.framework.ops.Tensor'> <dtype: 'int64'>
        preds_value = tf.expand_dims(b1,1)#Tensor("ExpandDims:0", shape=(?, 1), dtype=int64) (?, 1) <class 'tensorflow.python.framework.ops.Tensor'> <dtype: 'int64'>
        print("preds_indice:", preds_indice.shape)
        print("preds_values:",preds_value.shape)
        concated1 = tf.concat([preds_indice, preds_value], 1)#Tensor("concat_1:0", shape=(?, 2), dtype=int64) (?, 2) <class 'tensorflow.python.framework.ops.Tensor'> <dtype: 'int64'>
        labels_indice = tf.expand_dims(a2, 1)
        labels_value = tf.expand_dims(b2, 1)
        concated2 = tf.concat([labels_indice, labels_value], 1)
        '''使用tf.sparse_to_dense()时遇到的is out of order报错
        问题现象：运行上面程序，报is out of order错误。但如果把indices=tf.constant([[1,3],[1,2],[4,2]])
        换成indices=tf.constant([[1,3],[1,4],[4,2]])就运行正常。
        原因：在不设置参数validate_indices=False时，tf.sparse_to_dense要求indices必须是递增的。这个主要是为了方便函数检查indices是否有重复的。'''
        print("contact1:",concated1.shape)
        onehot_preds = tf.sparse_to_dense(concated1, tf.stack([batch_size, batch_size]), 1.0, 0.0,validate_indices=False)
        onehot_labels = tf.sparse_to_dense(concated2, tf.stack([batch_size, batch_size]), 1.0, 0.0,validate_indices=False)
        onehot_preds= create_symmetry(onehot_preds)
        onehot_labels = create_symmetry(onehot_labels)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=onehot_preds,labels=onehot_labels)
        loss2 = tf.reduce_mean(loss)

        return loss2


    def NoAdjLoss1(pred,labels,T1):
        # 直接做二值化，不是对称的
        labels = tf.argmax(labels, 1)#(10242,)
        preds = tf.argmax(pred, 1)#(10242,)
        batch_size = tf.cast(tf.reduce_max(labels)+1,dtype=tf.int32)#最大的数是35，但是这里的矩阵应该是36
        '''triangle_path = 'triangle/lhtriangle.txt'
        T1 = pd.read_csv(triangle_path, encoding='utf-8', header=None, sep=' ', names=['source', 'target'])
        T1 = T1.values'''

        def create_triangle_adj(triangle, batch_size, labels):
            new_tensor_list = []
            for i in range(triangle.shape[0]):
                m = triangle[i][0]
                n = triangle[i][1]
                m1 = labels[m]  # ()
                n1 = labels[n]
                new_tensor_list.append(m1)
                new_tensor_list.append(n1)

            # new_tensor_list   <class 'list'> 12
            new_tensor = tf.convert_to_tensor(new_tensor_list)  # (12,)
            new_tensor = tf.reshape(new_tensor, [triangle.shape[0], 2])
            new_tensor = tf.cast(new_tensor, dtype=tf.int32)  # (6,2)
            onehot_labels = tf.sparse_to_dense(new_tensor, tf.stack([batch_size, batch_size]), 1.0, 0.0,
                                               validate_indices=False)
            return onehot_labels
        triangle_label = create_triangle_adj(T1,batch_size,labels)
        triangle_pred =  create_triangle_adj(T1,batch_size,preds)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=triangle_pred,labels=triangle_label)
        loss2 = tf.reduce_mean(loss)

        return loss2



    def NoAdjLoss2(pred,labels,T1):
        #二值化之后做对称的
        labels = tf.argmax(labels, 1)#(10242,)
        preds = tf.argmax(pred, 1)#(10242,)
        batch_size = tf.cast(tf.reduce_max(labels)+1,dtype=tf.int32)#最大的数是35，但是这里的矩阵应该是36
        ####得到triangel_adj   pred和label
        def create_triangle_adj(triangle, batch_size, labels):
            new_tensor_list = []
            for i in range(triangle.shape[0]):
                m = triangle[i][0]
                n = triangle[i][1]
                m1 = labels[m]  # ()
                n1 = labels[n]
                new_tensor_list.append(m1)
                new_tensor_list.append(n1)

            # new_tensor_list   <class 'list'> 12
            new_tensor = tf.convert_to_tensor(new_tensor_list)  # (12,)
            new_tensor = tf.reshape(new_tensor, [triangle.shape[0], 2])
            new_tensor = tf.cast(new_tensor, dtype=tf.int32)  # (6,2)
            onehot_labels = tf.sparse_to_dense(new_tensor, tf.stack([batch_size, batch_size]), 1.0, 0.0,
                                               validate_indices=False)
            return onehot_labels

        triangle_label = create_triangle_adj(T1,batch_size,labels)
        triangle_pred =  create_triangle_adj(T1,batch_size,preds)

        ###对称
        def create_symmetry(data):
            m2 = data
            m3 = tf.transpose(m2)
            m4 = m3 + m2
            #下面的转换类型是自己改的
            #m4 = tf.cast(m4,dtype=tf.int32)
            m5 = tf.clip_by_value(m4, tf.convert_to_tensor([0.0]), tf.convert_to_tensor([1.0]))
            return m5
        triangle_label= create_symmetry(triangle_label)
        triangle_pred = create_symmetry(triangle_pred)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=triangle_pred,labels=triangle_label)
        loss2 = tf.reduce_mean(loss)

        return loss2


    def confusion_loss(logits, labels,nbclass):
        '''我感觉这里相当于把预测错误率作为了一个loss的意思，那交叉熵损失的含义何在？
        还是得使用那种用脑区约束来比较正确'''
        pred=tf.argmax(logits, 1)
        label=tf.argmax(labels, 1)
        A= tf.confusion_matrix(labels=label,predictions=pred,num_classes=nbclass)

        sum = tf.constant(0)
        digsum = tf.constant(0)
        for i in range(nbclass):
            for j in range(nbclass):
                sum += A[i][j]
        for i in range(nbclass):
            digsum += A[i][i]
        rlt = (sum - digsum) / sum
        rlt = tf.cast(rlt, dtype=tf.float32)
        print("result:",rlt.dtype,type(rlt))
        return rlt

    def dice_coff1(y_pred, y_true):
        A = tf.cast(tf.argmax(y_pred,1), dtype=tf.float32)#(10242,)
        B = tf.cast(tf.argmax(y_true,1), dtype=tf.float32)#(10242,)
        #A: Tensor("Cast_1:0", shape=(10242,), dtype=float32)
        #B: Tensor("Cast_2:0", shape=(10242,), dtype=float32)

        dice = 0.0
        for i in range(1, 36):
            A1 = tf.cast(tf.equal(A, tf.constant(i, dtype=tf.float32)), tf.float32)
            B1 = tf.cast(tf.equal(B, tf.constant(i, dtype=tf.float32)), tf.float32)
            #print("A1:",A1,A1.shape,"B1:",B1,B1.shape)
            #A1: Tensor("Cast_3:0", shape=(10242,), dtype=float32) (10242,) B1: Tensor("Cast_4:0", shape=(10242,), dtype=float32) (10242,)
            rlt1 = tf.reduce_sum(A1) + tf.reduce_sum(B1)
            rlt2 = tf.reduce_sum(tf.multiply(A1, B1))
            dice += (2.0 * rlt2) / (rlt1 + 0.0001)

        return  dice / 31

    def dice_coff2(y_pred, y_true):
        A = tf.cast(tf.argmax(y_pred, 1), dtype=tf.float32)  # (10242,)
        B = tf.cast(tf.argmax(y_true, 1), dtype=tf.float32)  # (10242,)
        dice = 0.0
        # A是y_true B是y_pred
        for i in range(1, 36):
            A1 = tf.cast(tf.equal(A, tf.constant(i, dtype=tf.float32)), tf.float32)
            A2 = tf.where(tf.equal(A, tf.constant(i, dtype=tf.float32)), A, A + 79)
            B1 = tf.cast(tf.equal(A2, B), tf.float32)

            rlt1 = tf.reduce_sum(A1) + tf.reduce_sum(B1)
            rlt2 = tf.reduce_sum(tf.multiply(A1, B1))
            dice += (2.0 * rlt2) / (rlt1 + 0.0001)
        return dice / 31


    def dice_coff3(y_pred, y_true):
        A = tf.cast(tf.argmax(y_pred,1), dtype=tf.float32)#(10242,)
        B = tf.cast(tf.argmax(y_true,1), dtype=tf.float32)#(10242,)

        dice = 0.0
        for i in range(1, 36):
            A1 = tf.cast(tf.equal(A, tf.constant(i, dtype=tf.float32)), tf.float32)
            B1 = tf.cast(tf.equal(B, tf.constant(i, dtype=tf.float32)), tf.float32)
            rlt1 = tf.reduce_sum(A1) + tf.reduce_sum(B1)
            rlt2 = tf.reduce_sum(tf.multiply(A1, B1))
            dice += (2.0 * rlt2) / (rlt1 + 0.0001)

        return  dice / 31



    def Hausdorff_distance(mask_gt,mask_pred):
        import surface_distance as surfdist
        print("mask_gt:",mask_gt.shape,type(mask_gt))
        print("mask_pred:", mask_pred.shape, type(mask_pred))
        '''mask_gt: (10242, 36) <class 'tensorflow.python.framework.ops.Tensor'>
           mask_pred: (10242, 36) <class 'tensorflow.python.framework.ops.Tensor'>'''

        surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
        hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
        '''compute_robust_hausdorff这个函数的第二个参数表示最大距离分位数，取值范围为0-100，它表示的是计算步骤4中，
        选取的距离能覆盖距离的百分比，例如我这里选取了95%，那么在计算步骤4中选取的不是最大距离，
        而是将距离从大到小排列后，取排名为5%的距离。这么做的目的是为了排除一些离群点所造成的不合理的距离，保持整体数值的稳定性。'''
        #avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
        #surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
        #surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)

        return hd_dist_95

    def create_symmetry(data):
        m2 = data
        m3 = tf.transpose(m2)
        print("m3:", m3, m3.shape, type(m3))
        m4 = m3 + m2
        print("m4:", m4, m4.shape, type(m4))

        ###进行二值化
        '''tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
        小于min的让它等于min，大于max的元素的值等于max。'''
        m5 = tf.clip_by_value(m4, tf.convert_to_tensor([0]), tf.convert_to_tensor([1]))
        return m5








































    '''    def dice_col_coffient1(y_pred, y_true):
        y_index = tf.where(y_true)
        y_pred1 = tf.nn.softmax(y_pred)

        y_true = tf.to_float(y_true)
        test1 = tf.multiply(y_pred1, y_true)

        zero = tf.constant(0, dtype=tf.float32)
        test2 = tf.not_equal(test1, zero)
        test2 = tf.boolean_mask(test1, test2)

        test3 = tf.to_float(y_index[:, 1])
        test4 = tf.stack([test3, test2], axis=1)

        #每个类别的个数
        test5 = test4[:, 0]
        test11 = tf.reverse((tf.nn.top_k(test5, k=10242).indices), axis=[0])
        test12 = tf.to_int32(tf.gather(test5, test11))
        list = []
        for i in range(36):
            res = tf.reduce_sum(tf.cast(tf.equal(test12, i), tf.int32))
            list.append(res)
        test13 = tf.convert_to_tensor(list)

        # 每个类别的概率和
        list1 = []
        zero1 = tf.constant(0, dtype=tf.int32)
        for i in range(36):
            test111 = tf.convert_to_tensor([tf.cast(tf.equal(test4[:, 0], i), tf.int32)])
            test21 = tf.not_equal(test111, zero1)  # [[False False False ... False False False]] (1, 10242)
            test22 = tf.to_int32(test21)
            test23 = tf.to_float(tf.reshape(test22, [10242, ]))  # (10242,)
            test24 = tf.multiply(test23, y_pred1[:, i])  # [ 0. -0. -0. ... -0.  0. -0.] (10242,)
            res = tf.reduce_sum(test24)
            list1.append(res)
        test14 = tf.convert_to_tensor(list1)  # (36,)

        dice_coff = 0.0
        for i in range(36):
            dice_index = 2 * (test14[i]) / (tf.to_float(test13[i]) + test14[i]+0.0000001)
            dice_coff += dice_index

        result = dice_coff / 36

        return result

    def dice_col_coffient2(y_pred, y_true):
        y_index = tf.where(y_true)
        y_pred1 = tf.nn.softmax(y_pred)
        test1 = tf.multiply(y_pred1, tf.to_float(y_true))
        test2 = tf.gather_nd(test1, y_index)
        test4 = tf.stack([tf.to_float(y_index[:, 1]), test2], axis=1)
        test11 = tf.reverse((tf.nn.top_k(test4[:, 0], k=10242).indices), axis=[0])
        test12 = tf.to_int32(tf.gather(test4[:, 0], test11))

        list = []
        list1 = []
        zero1 = tf.constant(0, dtype=tf.int32)
        for i in range(36):
            res = tf.reduce_sum(tf.cast(tf.equal(test12, tf.constant(i, dtype=tf.int32)), tf.int32))
            list.append(res)
            test111 = tf.convert_to_tensor([tf.cast(tf.equal(test4[:, 0], tf.constant(i, dtype=tf.float32)), tf.int32)])
            test21 = tf.not_equal(test111, zero1)
            test24 = tf.reduce_sum(tf.multiply(tf.to_float(tf.reshape(tf.to_int32(test21), [10242, ])),y_pred1[:, i]))
            list1.append(test24)

        test13 = tf.convert_to_tensor(list)
        test14 = tf.convert_to_tensor(list1)  # (36,)

        dice_coff = 0.0
        for i in range(36):
            dice_index = 2 * tf.abs((test14[i])) / tf.abs((test14[i] + tf.to_float(test13[i]) + 0.0000000000000001))
            dice_coff += dice_index

        result = 1.0-(dice_coff / 36)
        return result


    def dice_col_coffient3(y_pred, y_true):
        y_index = tf.where(y_true)
        y_pred1 = tf.nn.softmax(y_pred)
        test1 = tf.multiply(y_pred1, tf.to_float(y_true))
        test2=tf.gather_nd(test1, y_index)
        test4 = tf.stack([tf.to_float(y_index[:, 1]), test2], axis=1)
        test11 = tf.reverse((tf.nn.top_k(test4[:, 0], k=10242).indices), axis=[0])
        test12 = tf.to_int32(tf.gather(test4[:, 0], test11))

        list = []
        list1 = []
        zero1 = tf.constant(0, dtype=tf.int32)
        for i in range(36):
            res = tf.reduce_sum(tf.cast(tf.equal(test12, tf.constant(i, dtype=tf.int32)), tf.int32))
            list.append(res)
            test111 = tf.convert_to_tensor([tf.cast(tf.equal(test4[:, 0], tf.constant(i, dtype=tf.float32)), tf.int32)])
            # 不等于0   则为true=1
            test21 = tf.not_equal(test111, zero1)  # [[False False False ... False False False]] (1, 10242)
            test24 = tf.reduce_sum(tf.multiply(tf.to_float(tf.reshape(tf.to_int32(test21), [10242, ])) , y_pred1[:, i])) # [ 0. -0. -0. ... -0.  0. -0.] (10242,)
            list1.append(test24)

        # 升序排列  #(36,)  这就是0----35 每一个类别的节点数，也就是出现的次数
        test13 = tf.convert_to_tensor(list)
        # 每个类别的概率和
        test14 = tf.convert_to_tensor(list1)  # (36,)

        dice_coff = 0.0
        for i in range(36):
            #dice_index = 2 * (test14[i]) / (test14[i] + 1.0)
            #dice_index = 2 * tf.abs((test14[i])) / tf.abs((test14[i] + tf.to_float(test13[i]) / 36 + 0.000001))

            #相当于每个类别真实的概率都是1/36  不现实嘛
            dice_index = 2 * tf.abs((test14[i])) / tf.abs((test14[i] + tf.to_float(test13[i]) + 0.0000000000000001))
            dice_coff += dice_index

        result = dice_coff / 36
        return result
'''




















