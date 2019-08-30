# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

import os
import torch
import numpy as np


class CSVQADataset(Dataset):
    def __init__(self, path):
        datatypes = [('kb', 'S1000'), ('feature', 'S10'), ('value', 'S10'), ('question', 'S10'), ('answer', 'S10')]
        frame = np.loadtxt(path, dtype=datatypes, delimiter=';')
        self.frame = []

        for row in frame:
            # unpack input
            kb, feature, value, question, answer = row
            kb = torch.tensor(list(map(int, kb.split())))
            feature = torch.tensor(list(map(int, feature.split())))
            value = torch.tensor(list(map(int, value.split())))

            question = torch.tensor(list(map(int, question.split())))
            answer = torch.tensor(list(map(int, answer.split())))

            _input = (kb, feature, value, question)
            _output = (answer)
            self.frame.append((_input, _output))

    def get_n_features(self):
        return self.frame[0][0][0].size(0)

    def get_output_size(self):
        return [self.frame[0][0][1].size(0), self.frame[0][0][2].size(0)]

    def get_output_max(self):
        return max(x[0][2].item() for x in self.frame) + 1

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]



class DataCreator:
    def __init__(self, n_features, n_values):
        self.n_features = n_features
        self.n_values = n_values

    def generate_data(self, data_path, data_size, train_data_name='train.data', validation_data_name='validation.data'):
        try:
            os.stat(data_path)
        except:
            os.mkdir(data_path)

        # create dataset using numpy random function
        features = np.random.randint(self.n_features, size=data_size)
        questions = np.random.randint(self.n_features, size=data_size)
        values = np.random.randint(self.n_values, size=data_size)

        dataset = []
        for feat, val, question in zip(features, values, questions):
            kb = np.random.randint(self.n_values, size=self.n_features)
            
            if feat != question:
                answer = kb[question]
            else:
                answer = val

            value_tensor = torch.tensor(list([val]))
            feature_tensor = torch.tensor(list([feat]))
            kb_tensor = torch.tensor(kb)
            question_tensor = torch.tensor(list([question]))
            answer_tensor = torch.tensor(list([answer]))

            _input = (kb_tensor, feature_tensor, value_tensor)
            _output = (question_tensor, answer_tensor)
            dataset.append((_input, _output))

        train_size = int(data_size * 0.8)
        train_data = dataset[:train_size]
        self.write_data(train_data, os.path.join(data_path, train_data_name))

        validation_data = dataset[train_size:]
        self.write_data(validation_data, os.path.join(data_path, validation_data_name))

        return train_data, validation_data


    def write_data(self, data, file_path):
        with open(file_path, 'w') as f:
            for (kb, feature, value), (question, answer) in data:

                kb = ' '.join(str(int(x)) for x in kb.numpy())
                feature = ' '.join(str(int(x)) for x in feature.numpy())
                value = ' '.join(str(int(x)) for x in value.numpy())
                question = ' '.join(str(int(x)) for x in question.numpy())
                answer = ' '.join(str(int(x)) for x in answer.numpy())
                f.write(kb + ';' + feature + ';' + value + ';' + question + ';' + answer + '\n')




