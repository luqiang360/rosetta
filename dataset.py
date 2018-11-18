#!/usr/bin/env python

import tensorflow as tf

def _data_agument(image, name=None, stretch=1.2):
    '''
        Image Pre-processing without Data Agument
    '''
    with tf.name_scope(name, default_name='data_agument'):
        height, width = image.get_shape()

        if stretch:
            # A stretching factor of 1.2 leads to superior results 
            # than using the original aspect ratio
            width = int(1.2 * width)
            image = tf.image.resize_images(image, [height, width])

        new_width = int(32 * width / height)

        target_width = tf.placehoder(tf.int32, 'target_width')

        if new_width > target_width:
            image = tf.image.resize_images(image, [32, target_width])
        else:
            image = tf.image.resize_images(image, [32, new_width])
            image = tf.pad(image, tf.constant([[0, 0], [0, target_width - new_width]]))

        return image

def _record_parse(example, name=None):
    '''
        TODO
    '''
    with tf.name_scope(name, default_name='record_parse'):
        features = {
            'image': tf.FixedLenFeature((), dtype=tf.string), 
            'label': tf.FixedLenFeature((), dtype=tf.int32), 
            'length': tf.FixedLenFeature((), dtype=tf.int32)
        }

        parse = tf.parse_single_example(example, features)
        image = tf.image.convert_image_dtype(tf.image.decode_jpeg(parse['image']), tf.float32)
        image = _data_agument(image)

        return image, parse['label'], parse['length']

def _filter_word(example, name=None):
    '''
        TODO
    '''
    with tf.name_scope(name, default_name='filter_word'):
        length = tf.placehoder(tf.int32, 'length')

        if example[2] <= length:
            return True
        else:
            return False

def record_input(pattern, name=None, cycle_length=32, shuffle_buffer_size=1024, batch_size=64, num_cores=8, prefetch_buffer_size=4):
    '''
        TODO
    '''
    with tf.name_scope(name, default_name='record_input'):
        files = tf.data.Dataset.list_files(pattern)
        # parallelize the execution of and interleave the contents of data file readers
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length, sloppy=True))
        # shuffle and repeat a dataset returing a new permutation for each epoch
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size))
        # fuse the map and batch transformations, parallelize the batch creation
        dataset = dataset.apply(tf.contrib.data.map_and_batch(_record_parse, batch_size, num_parallel_batches=num_cores))
        dataset = dataset.filter(_filter_word)
        # prefetch elements from the input dataset ahead of the time they are requested
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        return dataset

if __name__ == '__main__':
    pass