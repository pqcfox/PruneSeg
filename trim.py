#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trims weights on a pruned model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import sys

import collections

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.analyze as ana
import tensorvision.utils as utils
import tensorvision.core as core 

from evaluation import kitti_test

flags.DEFINE_string('RUN', 'KittiSeg_pretrained',
                    'Modifier for model parameters.')
flags.DEFINE_string('hypes', 'hypes/KittiSeg.json',
                    'File storing model parameters.')
flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))


segmentation_weights_url = ("ftp://mi.eng.cam.ac.uk/"
                            "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, FLAGS.RUN)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not FLAGS.RUN == 'KittiSeg_pretrained':
        return

    import zipfile
    download_name = utils.download(segmentation_weights_url, runs_dir)

    logging.info("Extracting KittiSeg_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                'KittiSeg')
    else:
        runs_dir = 'RUNS'

    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)
    utils._add_paths_to_sys(hypes)
    train.maybe_download_and_extract(hypes)
    maybe_download_and_extract(runs_dir)

    logging.info("Trimming weights.")

    logdir = os.path.join(runs_dir, FLAGS.RUN)
    modules = utils.load_modules_from_logdir(logdir)

    with tf.Graph().as_default():

        # prepaire the tv session
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)
        image.set_shape([1, None, None, 3])
        inf_out = core.build_inference_graph(hypes, modules,
                                             image=image)

        # Create a session for running Ops on the Graph.
        shutil.copytree(logdir, 'trimmed')
        sess = tf.Session()
        saver = tf.train.Saver()
        core.load_weights('trimmed', sess, saver)
    
        for weight in tf.contrib.model_pruning.get_masks():
            print(weight.name)
            if weight.name in hypes['pruning']['layers']:
                weight_value = sess.run(weight)
                l1_values = np.sum(np.abs(weight_value), axis=[0, 1, 2])
                toss_kernels = l1_values.argsort()[:hypes['pruning']['m']]
                weight_value[:, :, :, toss_kernels] = 0
                assign_op = state_ops.assign(weight, tf.constant(weight_value))
                sess.run(assign_op)

        trim_output = os.path.join('trimmed', 'trimmed-' + hypes['pruning']['name'])
        os.mkdir(trim_output)
        saver.save(sess, trim_output)


if __name__ == '__main__':
    tf.app.run()
