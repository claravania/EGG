# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import argparse
import egg.core as core

from egg.zoo.common_ground.features import DataCreator


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data.train',
                        help='Name of the training data')
    parser.add_argument('--validation_data', type=str, default='data.dev',
                        help='Name of the validation data')

    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of posssible values.')
    parser.add_argument('--n_features', type=int, default=5,
                        help='Number of possible features.')

    parser.add_argument('--create_data', type=str, default='./data',
                        help='When set, we create new training and validation data, under the data directory.')
    parser.add_argument('--dataset_size', type=int, default=10000,
                        help='Only used if we choose to create data, we will use 80:20 split for training and validation.')

    args = core.init(parser)
    return args



if __name__ == "__main__":
    opts = get_params()

    data_creator = DataCreator(opts.n_features, opts.n_classes)
    train_data, validation_data = data_creator.generate_data(opts.create_data, opts.dataset_size, \
    	train_data_name=opts.train_data, validation_data_name=opts.validation_data)

    print('Generate dataset of size:', opts.dataset_size)