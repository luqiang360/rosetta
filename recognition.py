#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from image import record_input

def _get_variable(shape, trainable=True, scale=None):
    '''
        CPU Weight and Bias
    '''
    with tf.device('/device:CPU:0'):
        w = tf.get_variable('w', shape=shape, dtype=tf.float32, trainable=trainable, 
                initializer=tf.contrib.layers.xavier_initializer(), 
                regularizer=tf.contrib.layers.l2_regularizer(scale) if scale else None)
        b = tf.get_variable('b', shape=shape[-1:], dtype=tf.float32, 
                initializer=tf.constant_initializer(), trainable=trainable)

        return w, b

def _conv2d(x, filters, name=None, kernel=[3, 3], strides=[1, 2, 2, 1], padding='SAME'):
    '''
        2D Convolution
    '''
    with tf.variable_scope(name, default_name='conv'):
        shape = [kernel[0], kernel[1], x.get_shape()[-1], filters]
        w, b = _get_variable(shape)

        return tf.nn.bias_add(tf.nn.conv2d(x, w, strides, padding), b)

def _batch_norm(x, name=None, decay=0.9, epsilon=1e-5, is_training=True):
    '''
        Batch Norm
    '''
    with tf.variable_scope(name, default_name='bn'):
        return tf.contrib.layers.batch_norm(x, decay=decay, scale=True, 
                epsilon=epsilon, updates_collections=None, is_training=is_training)

def _relu(x, name=None):
    '''
        ReLU
    '''
    with tf.name_scope(name, default_name='relu'):
        return tf.nn.relu(x)

def _pool(x, name=None, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'):
    '''
        Max Pool
    '''
    with tf.name_scope(name, default_name='pool'):
        return tf.nn.max_pool(x, ksize, strides, padding)

def _residual(x, filters, name=None, is_training=True):
    '''
        Residual Block
    '''
    with tf.variable_scope(name, default_name='residual'):
        y = _relu(_batch_norm(x, is_training=is_training))
        y = _conv2d(y, filters, name='conv1')

        y = _relu(_batch_norm(y, is_training=is_training))
        y = _conv2d(y, filters, name='conv2', strides=[1, 1, 1, 1])

        return tf.add(x, y)

def _model(x, name=None, reuse=tf.AUTO_REUSE, is_training=True, alphabet=26):
    '''
        Text Recognition
    '''
    with tf.variable_scope(name, default_name='recognition', reuse=reuse):
        x = _conv2d(x, 64, name='conv1', kernel=[7, 7])
        x = _batch_norm(x, name='bn1', is_training=is_training)
        x = _pool(x, _relu(x))

        x = _residual(x,  64, name='residual1_1', is_training=is_training)
        x = _residual(x,  64, name='residual1_2', is_training=is_training)

        x = _residual(x, 128, name='residual2_1', is_training=is_training)
        x = _residual(x, 128, name='residual2_2', is_training=is_training)

        x = _residual(x, 256, name='residual3_1', is_training=is_training)
        x = _residual(x, 256, name='residual3_2', is_training=is_training)

        '''
            Remove the global average pooling layer at the end of the model 
            and replace the fully connected layer with a convolutional layer 
            that can accpet inputs of different lengths.
            Reduce the stride(1) of the last convolutional layers to better 
            preserve the spatial resolution of the features.
            The activation of the extra unit is the probability of observing 
            a blank character(alphabet + 1).
            Tensorflow's CTC loss performs the softmax operation.

            Output shape: [batch_size, number_of_character, 1, alphabet + 1]
        '''
        x = _relu(_conv2d(x, alphabet + 1, name='sequence', kernel=[2, 3], strides=[1, 1, 1, 1]))

        '''
            time_marjor == True(default)

            Output shape: [number_of_character, batch_size, alphabet + 1]
        '''
        return tf.transpose(tf.sequence(x, axis=[2]), perm=[1, 0, 2])

def _loss(labels, inputs, sequence_length, name=None):
    '''
        CTC Loss
    '''
    with tf.name_scope(name, default_name='ctc_loss'):
        '''
            labels shape: [batch_size, ]
            inputs shape: [number_of_character, batch_size, alphabet + 1]
            sequence_length shape: [batch_size]
        '''
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels, inputs, sequence_length))

        return loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

def _gradient(optimizer, loss, name=None, var_list=None):
    '''
        compute gradient
    '''
    with tf.name_scope(name, default_name='gradient'):
        return optimizer.compute_gradients(loss, var_list=var_list)

def _average(gradients, name=None):
    '''
        average gradient
    '''
    with tf.name_scope(name, default_name='average'):
        return [(tf.reduce_mean(tf.stack([y[0] for y in x]), axis=0), x[0][1]) for x in zip(*gradients)]

def _update(optimizer, grads_and_vars, name=None, global_step=None):
    '''
        update gradient
    '''
    with tf.name_scope(name, default_name='update'):
        return optimizer.apply_gradients(grads_and_vars, global_step=global_step)

def train(pattern, alpha, beta, N, M, l0, w0, step, decay, num_gpus=1, alphabet=26):
    '''
    '''
    with tf.Graph().as_default() as graph, tf.device('/device:CPU:0'):
        gradients = []

        with tf.variable_scope('CPU'):
            global_step = tf.train.create_global_step(graph=graph)
            learning_rate = tf.placehoder(tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            dataset = record_input(pattern)
            iterator = dataset.make_initializable_iterator()
            length = tf.get_tensor_by_name('length')
            target_width = tf.get_tensor_by_name('target_width')

        for i in range(num_gpus):
            with tf.device('/device:GPU:{}'.format(i)), tf.name_scope('GPU'):
                images, labels, sequence_length = iterator.get_next()
                inputs = _model(images, name='recognition', alphabet=alphabet)
                loss_op = _loss(labels, inputs, sequence_length)
                gradients.append(_gradient(optimizer, loss_op))

        with tf.name_scope('CPU'):
            gradient_op = _update(optimizer, _average(gradients), global_step=global_step)

            train_ops = tf.group(gradient_op)
            saver_ops = tf.train.Saver(var_list=tf.global_variables())
            init_ops = tf.global_variables_initializer()

        graph.finalize()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_ops)

        # Warmup training
        log_increment = (np.log(beta) - np.log(alpha)) / N
        lr = alpha
        l = l0

        for i in range(N):
            sess.run(iterator.initializer, feed_dict={length: l, target_width: w0})

            for j in range(step):
                _, loss_value = sess.run([train_ops, loss_op], feed_dict={learning_rate: lr})
                print('[Warmup training] epoch: {}, step: {}, loss: {}'.format(i, j, loss_value))

            l += 1
            lr += np.power(10, i * log_increment)

        # Post-warmup training
        lr = beta
        w = w0

        for i in range(M):
            sess.run(iterator.initializer, feed_dict={length: 128, target_width: w})

            for j in range(step):
                _, loss_value = sess.run([train_ops, loss_op], feed_dict={learning_rate: lr})
                print('[Post-warmup training] epoch: {}, step: {}, loss: {}'.format(i, j, loss_value))

            w += 8
            lr *= np.power(10, -int(i / decay))

if __name__ == '__main__':
    pass