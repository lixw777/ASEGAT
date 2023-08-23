import numpy as np
import tensorflow as tf
from models import SpGAT
from utils import process_mindboggle
import pandas as pd

"""将dice_loss(dice_3啥的)加入进去并且设置为0.5  """


'''对其多训练几次，效果就上来了'''
checkpt_file = 'pre_trained/data/mod_data.ckpt'
pathname = 'data/subject*'
num_subject = process_mindboggle.get_num_subjects(pathname)
sparse = True
batch_size = 1
nb_epochs = 100
patience = 5  # not change
lr = 0.002  # learning rate
l2_coef = 0.0002  # weight decay  lr/10
n_heads = [32,1]  # additional entry for the output layer
hid_units = [16]  # numbers of hidden units per each attention head in each layer



residual = False
nonlinearity = tf.nn.elu
model = SpGAT
alpha = 0.5
print("alpha:",alpha)

triangle_path = 'triangle/lhtriangle.txt'
T1 = pd.read_csv(triangle_path, encoding='utf-8', header=None, sep=' ', names=['source', 'target'])
T1 = T1.values

# model.summary()
print('nb_epochs:' + str(nb_epochs))
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))


# 由于adj都是一样的所以
edge_path = 'data/subject1/lhedges.txt'
biases = process_mindboggle.generate_adj(edge_path)
features_data, labels_data = process_mindboggle.generate_3D_data_sparse(num_subject)


nb_nodes = features_data.shape[1]
ft_size = features_data.shape[2]
nb_classes = labels_data.shape[2]


mask = (np.array(np.ones((batch_size, nb_nodes)), dtype=np.bool))
x_train, x_val, x_test, y_val, y_train, y_test, test_list = process_mindboggle.split_data_sparse(
    num_subject, features_data, labels_data, nb_nodes)
print("test1:",x_train.shape,type(x_train),x_test.shape,y_train.shape,y_test.shape )

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.sparse_placeholder(dtype=tf.float32)
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                             attn_drop, ffd_drop,
                             bias_mat=bias_in,
                             hid_units=hid_units, n_heads=n_heads,
                             residual=residual, activation=nonlinearity)  # logits: (1, 10242, 36)
    log_resh = tf.reshape(logits, [-1, nb_classes])  # log_resh: (10242, 36)
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])  # lab_resh: (10242, 36)
    msk_resh = tf.reshape(msk_in, [-1])
    loss1 = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    ACLoss = model.NoAdjLoss2(log_resh, lab_resh,T1)
    loss = loss1+alpha*ACLoss

    dice_loss = model.dice_coff3(y_pred=log_resh, y_true=lab_resh)


    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    train_op = model.training(loss, lr, l2_coef)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    vdice_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(nb_epochs):
            train_step = 0
            train_size = x_train.shape[0]  # 62
            train_loss_avg = 0
            train_acc_avg = 0
            train_dice_avg = 0
            while train_step * batch_size < train_size:
                _, loss_value_tr, acc_tr,dice_tr = sess.run([train_op, loss, accuracy,dice_loss],
                                                    feed_dict={
                                                        ftr_in: x_train[
                                                                train_step * batch_size:(train_step + 1) * batch_size],
                                                        bias_in: biases,
                                                        lbl_in: y_train[
                                                                train_step * batch_size:(train_step + 1) * batch_size],
                                                        msk_in: mask,
                                                        is_train: True,
                                                        attn_drop: 0.9, ffd_drop: 0.9})

                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                train_dice_avg += dice_tr
                train_step += 1

            val_loss_avg = 0
            val_acc_avg = 0
            val_dice_avg = 0
            val_step = 0
            val_size = x_val.shape[0]  # 8
            while val_step * batch_size < val_size:
                loss_value_val, acc_val, dice_val = sess.run([loss, accuracy,dice_loss],
                                                   feed_dict={
                                                       ftr_in: x_val[val_step * batch_size:(val_step + 1) * batch_size],
                                                       bias_in: biases,
                                                       lbl_in: y_val[val_step * batch_size:(val_step + 1) * batch_size],
                                                       msk_in: mask,
                                                       is_train: False,
                                                       attn_drop: 0.0, ffd_drop: 0.0})
                # NonAdjLoss = base_gattn.BaseGAttN.NoAdjLoss(pred_tr, y_train[0])
                val_loss_avg += loss_value_val
                val_acc_avg += acc_val
                val_dice_avg += dice_val
                val_step += 1

            print('epoch=%d|Training: loss = %.5f, acc = %.5f, dice_coff = %.5f | Val: loss = %.5f, acc = %.5f, dice_coff = %.5f' %
                  (epoch, train_loss_avg / train_step, train_acc_avg / train_step, train_dice_avg/train_step,
                   val_loss_avg / val_step, val_acc_avg / val_step, val_dice_avg/val_step))

            # 先满足一个条件，再满足两个条件
            if val_dice_avg / val_step >= vdice_mx or val_loss_avg / val_step <= vlss_mn:
                if val_dice_avg / val_step >= vdice_mx and val_loss_avg / val_step <= vlss_mn:
                    vdice_early_model = val_dice_avg / val_step
                    vlss_early_model = val_loss_avg / val_step
                    saver.save(sess, checkpt_file)
                vdice_mx = np.max((val_dice_avg / val_step, vdice_mx))
                vlss_mn = np.min((val_loss_avg / val_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max dice: ', vdice_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', dice: ',vdice_early_model)
                    break



            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = x_test.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0
        ts_dice = 0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts, pred_ts, dice_ts = sess.run([loss, accuracy, log_resh,dice_loss],
                                                      feed_dict={
                                                          ftr_in: x_test[
                                                                  ts_step * batch_size:(ts_step + 1) * batch_size],
                                                          bias_in: biases,
                                                          lbl_in: y_test[
                                                                  ts_step * batch_size:(ts_step + 1) * batch_size],
                                                          msk_in: mask,
                                                          is_train: False,
                                                          attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_dice += dice_ts

            pred_ts1 = tf.argmax(pred_ts, 1)
            pred_ts1 = sess.run(pred_ts1)
            np.savetxt('pred/pred_ts' + str(test_list[ts_step]) + '.txt', pred_ts1, fmt='%d')

            ts_step += 1


        # 这里就不需要求平均了，没意思了
        print('Test loss:', ts_loss/ts_step, '; accuracy:',ts_acc/ts_step,'; dice_coff:',ts_dice/ts_step)


    sess.close()
