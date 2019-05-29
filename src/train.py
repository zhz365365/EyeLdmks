import tensorflow as tf
import argparse
import input_dataset
import os
import numpy as np
import math
import time
import random
import cv2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--GPU', default='7', help='the index of gpu')
parser.add_argument('--batch_size', default=16, help='the size of examples in per batch')
parser.add_argument('--epoch', default=200, help='the train epoch')
parser.add_argument('--lr_boundaries', default='8,16,24,32,40,48,56', help='the boundaries of learning rate')
parser.add_argument('--lr_values', default='0.0004,0.00004,0.000004,0.0000004,0.00000004,0.000000004,0.0000000004,0.00000000004', help='the values of learning_rate')
parser.add_argument('--data_type', default='UnityEyes', help='the silent or move')
parser.add_argument('--data_dir', default='../dataset/', help='the directory of training data')
parser.add_argument('--server', default='77', help='server')
parser.add_argument('--net_name', default='resnet_v2', help='the name of the network')
parser.add_argument('--image_height', default='160', help='the down scale of image')
parser.add_argument('--image_width', default=256, help='the width of image')
parser.add_argument('--dropout_keep_prob', default=0.5, help='the probility to keep dropout')
parser.add_argument('--restore_step', default='0', help='the step used to restore')
parser.add_argument('--trainable', default='0', help='train or not')
parser.add_argument('--reduce_mean', default='0', help='preprocess of image')
parser.add_argument('--stack', default='3', help='the number of stacked hourglasses')
parser.add_argument('--activation_cross_entropy', default='sigmoid', help='the activity function')
parser.add_argument('--use_batch_norm', default='1', help='whether or not use BN')
args = parser.parse_args()
args.data_dir = args.data_dir + args.data_type
args.dropout_keep_prob = float(args.dropout_keep_prob)
args.reduce_mean = bool(args.reduce_mean == '1')
args.use_batch_norm = bool(args.use_batch_norm == '1')
args.stack = int(args.stack)
args.batch_size = int(args.batch_size)
args.epoch = int(args.epoch)
args.image_height = int(args.image_height)
args.image_width = int(args.image_width)
lr_boundaries = []
for key in args.lr_boundaries.split(','): 
    lr_boundaries.append(float(key))
lr_values = []
for key in args.lr_values.split(','):
    lr_values.append(float(key))
if args.net_name.startswith('hour'):
    import hourglasses as Gaze_Scale_Net
else:
    raise Exception('wrong net name!')

summary_dir = '../Log/' + args.server + '/event_log'
restore_dir = '../Log/' + args.server + '/check_log'
checkpoint_dir = '../Log/' + args.server + '/check_log/model.ckpt'
draw_dir = '../Log/' + args.server + '/draw_log'

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
slim = tf.contrib.slim

def arch_net(image, is_training, dropout_keep_prob):
    Net = Gaze_Scale_Net.Net(
        net_input=image,
        net_name=args.net_name,
        is_training=is_training,
        dropout_keep_prob=dropout_keep_prob,
        stack=args.stack,
		use_batch_norm=args.use_batch_norm,
    )

    for key in Net.end_points.keys():
        if isinstance(Net.end_points[key], dict):
            for sub_key in Net.end_points[key].keys():
                print(key + '/' + sub_key, Net.end_points[key][sub_key].get_shape().as_list())
        else:
            print(key, ' ', Net.end_points[key].get_shape().as_list())
    return Net.end_points[Net.scope + '/logits'], Net.end_points

def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    variables_to_train = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                print(var.op.name)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore, variables_to_train

class compute(object):
    def __init__(self, labels, ldmks, all_logits):
        self.loss = tf.Variable(0, dtype=tf.float32)
        self.labels = labels
        self.ldmks = ldmks
        for i in range(args.stack):
            logits = all_logits['P%d' % (4-i)]
            sub_loss = self._loss(logits)
            self.loss = self.loss + tf.reduce_mean(sub_loss, axis=0)

        logits = all_logits['P4']
        predictions_1 = []
        predictions_2 = []
        for i in range(logits.get_shape().as_list()[-1]):
            sub_logits = logits[:,:,:,i]
            sub_logits = tf.reshape(sub_logits, [-1, (args.image_height//4)*(args.image_width//4)])
            sub_logits = tf.nn.sigmoid(sub_logits)
            """
            arg_1_max
            """
            # [b]
            sub_predictions_1 = tf.arg_max(sub_logits, 1)
            predictions_1.append(sub_predictions_1)
            """
            arg_2_max
            """
            sub_predictions_2 = tf.reshape(tf.nn.top_k(sub_logits, 2)[1], [args.batch_size, 2])
            predictions_2.append(sub_predictions_2)
            """
            soft-argmax
            """
            # [hw]
            # index = tf.range(0, (args.image_height//4)*(args.image_width//4), 1, dtype=tf.float32) 
            # [b, hw]
            # index = tf.stack([index for i in range(args.batch_size)], axis=0)
            # [b, hw]
            # sub_logits = tf.exp(sub_logits)
            # [b]
            # sub_logits_sum = tf.reduce_sum(sub_logits, axis=1)
            # [b, hw]
            # sub_logits_sum = tf.stack([sub_logits_sum for i in range((args.image_height//4)*(args.image_width//4))], axis=1)
            # [b]
            #sub_predictions = tf.round(tf.reduce_sum(sub_logits/sub_logits_sum*index, axis=1))

        # [55, b]
        predictions_1 = tf.stack(predictions_1, axis=0)
        # [b, 55]
        predictions_1 = tf.transpose(predictions_1, [1, 0])
        predictions_row_1 = tf.cast(predictions_1 // (args.image_width//4), tf.float32)
        predictions_col_1 = tf.cast(predictions_1 % (args.image_width//4), tf.float32)
        # [b, 55, 1]
        predictions_row_1 = tf.reshape(predictions_row_1, [args.batch_size, 55, 1])
        predictions_col_1 = tf.reshape(predictions_col_1, [args.batch_size, 55, 1])
        # [b, 55, 2]
        self.predictions_1 = tf.concat([predictions_col_1, predictions_row_1], axis=2)

        # [55, b, 2]
        predictions_2 = tf.stack(predictions_2, axis=0)
        # [b, 55, 2]
        predictions_2 = tf.transpose(predictions_2, [1, 0, 2])
        predictions_row_2 = tf.cast(predictions_2 // (args.image_width//4), tf.float32)
        predictions_col_2 = tf.cast(predictions_2 % (args.image_width//4), tf.float32)
        smooth = 0.75
        # [b, 55, 1]
        predictions_row_2 = tf.reshape(predictions_row_2[:,:,0]*smooth+predictions_row_2[:,:,1]*(1-smooth), [args.batch_size, 55, 1])
        predictions_col_2 = tf.reshape(predictions_col_2[:,:,0]*smooth+predictions_col_2[:,:,1]*(1-smooth), [args.batch_size, 55, 1])
        # [b, 55, 2]
        self.predictions_2 = tf.concat([predictions_col_2, predictions_row_2], axis=2)

        self._error()

    def _loss(self, logits):
        shape = logits.get_shape().as_list()
        logits = tf.reshape(tf.transpose(logits, [0, 3, 1, 2]), [-1, shape[1]*shape[2]])
        labels = tf.reshape(tf.transpose(self.labels, [0, 3, 1, 2]), [-1, shape[1]*shape[2]])
        if args.activation_cross_entropy == 'softmax':
            return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        elif args.activation_cross_entropy == 'sigmoid':
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1)
        else:
            raise Exception('wrong activation cross entropy function!')
	
    def _error(self):
        self.place_error_1 = tf.sqrt(tf.reduce_mean(tf.pow(self.ldmks-self.predictions_1, 2)))
        self.place_error_2 = tf.sqrt(tf.reduce_mean(tf.pow(self.ldmks-self.predictions_2, 2)))

def visual_predictions(img_batch, ldmks_batch, predictions_1_batch, predictions_2_batch, epoch):
    num = random.randint(0, args.batch_size-1)
    img = (img_batch[num]*255.0)[:,:,::-1]; ldmks = ldmks_batch[num]; predictions_1 = predictions_1_batch[num]; predictions_2 = predictions_2_batch[num];
    ldmks = ldmks*4.0; predictions_1 = predictions_1*4.0; predictions_2 = predictions_2*4.0
    img_ldmks = img.copy(); img_predictions_1 = img.copy(); img_predictions_2 = img.copy()
    for ldmk in ldmks:
        cv2.circle(img_ldmks, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    for prediction_1 in predictions_1:
        cv2.circle(img_predictions_1, (int(prediction_1[0]), int(prediction_1[1])), 2, (255, 0, 0), -1)
    for prediction_2 in predictions_2:
        cv2.circle(img_predictions_2, (int(prediction_2[0]), int(prediction_2[1])), 2, (255, 0, 0), -1)
    cv2.imwrite(draw_dir+'/%.4f_ldmks.jpeg'%epoch, img_ldmks)
    cv2.imwrite(draw_dir+'/%.4f_predictions_1.jpeg'%epoch, img_predictions_1)
    cv2.imwrite(draw_dir+'/%.4f_predictions_2.jpeg'%epoch, img_predictions_2)

def visual_predictions_test(img_batch, predictions_1_batch, predictions_2_batch, epoch):
    num = random.randint(0, args.batch_size-1)
    img = (img_batch[num]*255.0)[:,:,::-1]; predictions_1 = predictions_1_batch[num]; predictions_2 = predictions_2_batch[num];
    predictions_1 = predictions_1*4.0; predictions_2 = predictions_2*4.0
    img_predictions_1 = img.copy(); img_predictions_2 = img.copy()
    for prediction_1 in predictions_1:
        cv2.circle(img_predictions_1, (int(prediction_1[0]), int(prediction_1[1])), 2, (255, 0, 0), -1)
    for prediction_2 in predictions_2:
        cv2.circle(img_predictions_2, (int(prediction_2[0]), int(prediction_2[1])), 2, (255, 0, 0), -1)
    cv2.imwrite(draw_dir+'/%.4f_predictions_test_1.jpeg'%epoch, img_predictions_1)
    cv2.imwrite(draw_dir+'/%.4f_predictions_test_2.jpeg'%epoch, img_predictions_2)

def train():

    print(tf.__version__)

    with tf.Graph().as_default():

        Data = input_dataset.Dataset_reader(data_type=args.data_type,
											batch_size=args.batch_size,
                                            image_height=args.image_height,
                                            image_width=args.image_width)

        image = tf.placeholder(tf.float32, [args.batch_size, args.image_height, args.image_width, 3])
        ldmks = tf.placeholder(tf.float32, [args.batch_size, 55, 2])
        labels = tf.placeholder(tf.float32, [args.batch_size, args.image_height//4, args.image_width//4, 55])
        dropout_keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)

        all_logits, end_points = arch_net(image, is_training, dropout_keep_prob)
        
        variables_to_restore,variables_to_train = g_parameter(args.net_name)

        g = compute(labels, ldmks, all_logits)
        slim.losses.add_loss(g.loss)
        total_loss = slim.losses.get_total_loss()

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
        num_epoch = tf.get_variable('num_epoch', [], trainable=False, dtype=tf.float32)
        num_epoch = tf.cast(global_step, tf.float32) * args.batch_size / (Data.train_nums)
        lr = tf.train.piecewise_constant(num_epoch, boundaries=lr_boundaries, values=lr_values)
        with tf.name_scope('loss'):
            tf.summary.scalar('loss', g.loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('learning_rate', lr)
        
        with tf.name_scope('error'):
            tf.summary.scalar('place_error_1', g.place_error_1)
            tf.summary.scalar('place_error_2', g.place_error_2)
        
        var_list = variables_to_train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            # Adam
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, var_list=var_list, global_step = global_step, name='Adam')
            # SGD
            # train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss, var_list=var_list, global_step=global_step, name='GD')
        saver_list = tf.global_variables()

        init = tf.global_variables_initializer()
        saver_restore = tf.train.Saver(saver_list)
        saver_train = tf.train.Saver(saver_list, max_to_keep = 100)
        merged = tf.summary.merge_all()

        def train_once(sess, summary_writer_train):
            image_batch, labels_batch, ldmks_batch = sess.run([Data.train_batch[0], Data.train_batch[1], Data.train_batch[2]])
            feed_dict_train = {image:image_batch, labels:labels_batch, ldmks:ldmks_batch, is_training:True, dropout_keep_prob:args.dropout_keep_prob}
            start_time = time.time()
            summary, loss_value, step, epoch, _ = sess.run([merged,
                                                            g.loss,
                                                            global_step,
                                                            num_epoch,
                                                            train_op],
                                                            feed_dict=feed_dict_train)
            end_time = time.time()
            sec_per_batch = end_time - start_time
            examples_per_sec = float(args.batch_size) / sec_per_batch
            summary_writer_train.add_summary(summary, int(1000 * epoch))
            print('epoch %d step %d with loss %.4f(%.1f examples/sec; %.3f sec/batch)' % (int(epoch), step, loss_value, examples_per_sec, sec_per_batch))
            return epoch, step

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer_train = tf.summary.FileWriter(logdir=summary_dir + '/train', graph=sess.graph)
            summary_writer_eval = tf.summary.FileWriter(logdir=summary_dir + '/eval')

            ckpt = tf.train.get_checkpoint_state(restore_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if args.restore_step == '0':
                    temp_dir = ckpt.model_checkpoint_path
                else:
                    temp_dir = ckpt.model_checkpoint_path.split('-')[0] + '-' + args.restore_step
                temp_step = int(temp_dir.split('-')[1])
                print('Restore the global parameters in the step %d!' % (temp_step)) 
                saver_restore.restore(sess, temp_dir)
            else:
                print('Initialize the global parameters')
                init.run()
 
            eval_epoch = 0
            test_epoch = 0 
            checkpoint_epoch = 0

            while(args.trainable == '1'):
                epoch, step = train_once(sess, summary_writer_train)
                if epoch > eval_epoch + 0.0005:
                    image_batch, labels_batch, ldmks_batch = sess.run([Data.eval_batch[0], Data.eval_batch[1], Data.eval_batch[2]])
                    image_test_batch, gaze_test_batch = sess.run([Data.test_batch[0], Data.test_batch[1]])
                    feed_dict_eval = {image:image_batch, labels:labels_batch, ldmks:ldmks_batch, is_training:False, dropout_keep_prob:1.0}
                    feed_dict_test = {image:image_test_batch, is_training:False, dropout_keep_prob:1.0}
                    ldmks_value, predictions_1_value, predictions_2_value, summary, loss_value = sess.run([g.ldmks, g.predictions_1, g.predictions_2, merged, g.loss], feed_dict=feed_dict_eval)
                    predictions_1_test_value, predictions_2_test_value = sess.run([g.predictions_1, g.predictions_2], feed_dict=feed_dict_test)
                    visual_predictions(image_batch, ldmks_value, predictions_1_value, predictions_2_value, epoch)
                    visual_predictions_test(image_test_batch, predictions_1_test_value, predictions_2_test_value, epoch)
                    summary_writer_eval.add_summary(summary, (1000 * epoch))
                    eval_epoch = epoch
                if epoch >= test_epoch + 1:
                    place_error_1 = []
                    place_error_2 = []
                    for i in range(int(Data.eval_nums / args.batch_size)):
                        image_batch, labels_batch, ldmks_batch = sess.run([Data.eval_batch[0], Data.eval_batch[1], Data.eval_batch[2]])
                        feed_dict_eval = {image:image_batch, labels:labels_batch, ldmks:ldmks_batch, is_training:False, dropout_keep_prob:1.0}
                        place_error_value_1, place_error_value_2 = sess.run([g.place_error_1, g.place_error_2], feed_dict=feed_dict_eval)
                        place_error_1.append(place_error_value_1)
                        place_error_2.append(place_error_value_2)

                    eval_dataset_place_error_1 = np.mean(np.array(place_error_1))
                    eval_dataset_place_error_2 = np.mean(np.array(place_error_2))

                    format_str = 'epoch %.3f step %d eval_dataset_place_error_%d = %.4f'
                    f = open(draw_dir + '/evaluation.txt', 'a')
                    print(format_str % (epoch, step, 1, eval_dataset_place_error_1), file=f)
                    print(format_str % (epoch, step, 2, eval_dataset_place_error_2), file=f)
                    f.close()
                    test_epoch = epoch
                if epoch > checkpoint_epoch + 0.1:
                    saver_train.save(sess, checkpoint_dir, global_step=step)
                    checkpoint_epoch = epoch
                if epoch >= args.epoch:
                    break
                
            summary_writer_train.close()
            summary_writer_eval.close()
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    print ("-----------------------------train.py start--------------------------")
    train()
