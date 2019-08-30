   # Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import os
import sys
import argparse
import contextlib

import torch.utils.data
import torch.nn.functional as F
import egg.core as core
import numpy as np

import egg.zoo.common_ground.archs as archs
from egg.zoo.common_ground.features import CSVQADataset
from egg.zoo.common_ground.archs import EarlyStopperAccuracy
from torch.utils.data import DataLoader



def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default=None,
                        help='Path for the train data')
    parser.add_argument('--validation_data', type=str, default=None,
                        help='Path for the validation data')
    parser.add_argument('--data_prefix', type=str, default=None,
                        help='If train and dev have the same prefix, can use this param instead.')
    
    parser.add_argument('--dump_data', type=str, default=None,
                        help='Path for the data to be dumped')
    parser.add_argument('--dump_output', type=str, default=None,
                        help='Path for the dump')

    parser.add_argument('--batches_per_epoch', type=int, default=100,
                        help='Number of batches per epoch (default: 100)')

    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Dimension of the embedding of feature and value (default: 100)')
    parser.add_argument('--hidden_dim', type=int, default=250,
                        help='Size of the hidden layer of Agent (default: 250)')
    parser.add_argument('--agent_type', type=str, default='fixed',
                        help='fixed or symmetric agent')
    parser.add_argument('--action_dim', type=int, default=10,
                        help='Dimension of the flag action (fact agent or question agent) (default: 10)')

    parser.add_argument('--fa_entropy_coeff', type=float, default=0.1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-2)')
    parser.add_argument('--qa_entropy_coeff', type=float, default=0.1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-2)')

    parser.add_argument('--read_function', type=str, default='concat',
                        help='How the sender will read its KB. Available options: concat100, concat125, concat175 (default: concat175)')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Value if we use weight decay (default: 0)')
    parser.add_argument('--early_stopping', type=int, default=500,
                        help='Use early stopping, stop after development accuracy does not improve for #threshold epochs.')


    args = core.init(parser, params)
    return args



def dump(opts, game, dataset, device):
    train_state = game.training
    game.eval()

    device = device if device is not None else common_opts.device

    fact_inputs, messages, question_feats, question_values = [], [], [], []
    gold_values = []

    # process data that will be evaluated
    with torch.no_grad():
        for batch in dataset:

            if game.agent_type == 'symmetric':
                agentA_start = bool(np.random.binomial(1, 0.5, size=1))
            else:
                agentA_start = True
            if agentA_start:
                fAgent, qAgent = game.agentA, game.agentB
                fAnswer, qAnswer = game.answerModA, game.answerModB
            else:
                fAgent, qAgent = game.agentB, game.agentA
                fAnswer, qAnswer = game.answerModB, game.answerModA

            # each batch is (fact_tuple, question_tuple)
            # fact_tuple is (kb, fact_feat, fact_value)
            # question_tuple is (ans_feat, ans_value)
            fact_input = core.util.move_to(batch[0], device)
            gold_value = core.util.move_to(batch[1], device)

            # start with a dummy message, index 0
            batch_size = fact_input[0].size(0)
            message = torch.zeros([batch_size], dtype=torch.int64).to(device)

            # action flag: 0 is for fact agent and 1 is for question agent
            fa_flag = torch.zeros([batch_size], dtype=torch.int64).to(device)
            qa_flag = torch.ones([batch_size], dtype=torch.int64).to(device)

            message, _, _, _ = fAgent(message, fa_flag, fact_input, fAgent=True, qAgent=False)

            # to test whether QA listens
            # perm = torch.randperm(message.size(0))
            # message = message[perm]
            
            _, state, _, _ = qAgent(message, qa_flag, fact_input, fAgent=False, qAgent=True)

            answer = qAnswer(state)

            if isinstance(fact_input, list) or isinstance(fact_input, tuple):
                fact_inputs.extend(zip(*fact_input))
            else:
                fact_inputs.extend(fact_input)

            messages.extend(message)
            question_feats.extend(answer['question_feat'])
            question_values.extend(answer['question_value'])
            gold_values.extend(gold_value)
        
    game.train(mode=train_state)
    print(opts)

    # PREDICTION
    question_acc = 0.0
    n_data = 0
    for fact_input, message, q_feat, q_value, gold_value in \
        zip(fact_inputs, messages, question_feats, question_values, gold_values):

        kb, fact_feature, fact_value, question = fact_input

        kb = ' '.join(map(str, kb.tolist()))
        fact_feature = fact_feature.tolist()[0]
        fact_value = fact_value.tolist()[0]
        question = question.tolist()[0]
        gold_value = gold_value.tolist()[0]
        
        message = message.tolist()

        q_feat = q_feat.argmax().tolist()
        q_value = q_value.argmax().tolist()

        if question == q_feat and gold_value == q_value:
            question_acc += 1.0

        n_data += 1
        
        
        print(f'{kb};{fact_feature};{fact_value};{message};{question};{gold_value};{q_feat};{q_value}')

    print("Question accuracy: ", question_acc * 100 / n_data)


def differentiable_loss(_sender_input, _message, _receiver_input, receiver_output, labels):

    # answer accuracy
    question_value_labels = labels.squeeze(1)
    question_feat_labels = _sender_input[3].squeeze(1)
    question_feat_acc = (receiver_output['question_feat'].argmax(dim=1) == question_feat_labels).detach().float().mean()
    question_value_acc = (receiver_output['question_value'].argmax(dim=1) == question_value_labels).detach().float().mean()

    # retrieve fact accuracy
    fact_feat_labels = _sender_input[1].squeeze(1)
    fact_value_labels = _sender_input[2].squeeze(1)
    fact_feat_acc = (receiver_output['fact_feat'].argmax(dim=1) == fact_feat_labels).detach().float().mean()
    fact_value_acc = (receiver_output['fact_value'].argmax(dim=1) == fact_value_labels).detach().float().mean()

    qfeat_loss = F.cross_entropy(receiver_output['question_feat'], question_feat_labels, reduction="none")
    qvalue_loss = F.cross_entropy(receiver_output['question_value'], question_value_labels, reduction="none")
    ffeat_loss = F.cross_entropy(receiver_output['fact_feat'], fact_feat_labels, reduction="none")
    fvalue_loss = F.cross_entropy(receiver_output['fact_value'], fact_value_labels, reduction="none")

    total_loss = qfeat_loss + qvalue_loss

    return total_loss, {'question_acc': question_feat_acc * question_value_acc, 'question_feat_acc': question_feat_acc, \
                        'question_value_acc': question_value_acc}
   


def build_model(opts, train_loader, dump_loader):
    n_features = train_loader.dataset.get_n_features() if train_loader else dump_loader.dataset.get_n_features()
    n_values = train_loader.dataset.get_output_max() if train_loader else dump_loader.dataset.get_output_max()

    receiver_outputs = {'feat':n_features, 'value':n_values}

    senderA = archs.SenderModule(opts.hidden_dim, opts.embedding_dim, opts.action_dim, opts.vocab_size)
    receiverA = archs.ReceiverModule(opts.hidden_dim, opts.vocab_size)
    agentA = archs.Agent(senderA, receiverA, n_features, n_values, opts)
    agentA = archs.ReinforceWrapper(agentA)

    senderB = archs.SenderModule(opts.hidden_dim, opts.embedding_dim, opts.action_dim, opts.vocab_size)
    receiverB = archs.ReceiverModule(opts.hidden_dim, opts.vocab_size)
    agentB = archs.Agent(senderB, receiverB, n_features, n_values, opts)
    agentB = archs.ReinforceWrapper(agentB)

    loss = differentiable_loss
    
    return agentA, agentB, loss


def main(params):
    opts = get_params(params)

    print(f'Launching game with parameters: {opts}')

    device = torch.device("cuda" if opts.cuda else "cpu")
    train_loader = None
    validation_loader = None
    dump_loader = None

    
    # load from given train and validation data
    if opts.data_prefix:
        train_data = opts.data_prefix + '.train'
        validation_data =  opts.data_prefix + '.dev'

        train_loader = DataLoader(CSVQADataset(path=train_data),
                              batch_size=opts.batch_size,
                              shuffle=False, num_workers=1)

        validation_loader = DataLoader(CSVQADataset(path=validation_data),
                                   batch_size=opts.batch_size,
                                   shuffle=False, num_workers=1)

    elif opts.train_data and opts.validation_data:
        train_data = opts.train_data
        validation_data = opts.validation_data

        train_loader = DataLoader(CSVQADataset(path=opts.train_data),
                              batch_size=opts.batch_size,
                              shuffle=False, num_workers=1)

        validation_loader = DataLoader(CSVQADataset(path=opts.validation_data),
                                   batch_size=opts.batch_size,
                                   shuffle=False, num_workers=1)


    # evaluate model on dump data
    if opts.dump_data:
        dump_loader = DataLoader(CSVQADataset(path=opts.dump_data),
                                 batch_size=opts.batch_size,
                                 shuffle=False, num_workers=1)
        

    assert train_loader or dump_loader, 'Either training or dump data must be specified'

    agentA, agentB, loss = build_model(opts, train_loader, dump_loader)
    game = archs.SymbolGameReinforce(agentA, agentB, loss, opts.agent_type, fa_entropy_coeff=opts.fa_entropy_coeff, qa_entropy_coeff=opts.qa_entropy_coeff)
    
    optimizer = torch.optim.Adam(game.parameters(), opts.lr, weight_decay=opts.weight_decay)
    early_stopper = EarlyStopperAccuracy(opts.early_stopping)

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, print_train_loss=True,
                           early_stopping=early_stopper)

    if dump_loader is not None:
        if opts.dump_output:
            with open(opts.dump_output, 'w') as f, contextlib.redirect_stdout(f):
                dump(opts, game, dump_loader, device)
        else:
            dump(opts, game, dump_loader, device)
    else:
        trainer.train(n_epochs=opts.n_epochs)
        if opts.checkpoint_dir:
            trainer.save_checkpoint()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    

