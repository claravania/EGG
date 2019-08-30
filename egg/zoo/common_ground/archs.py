# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import egg.core as core

from torch.distributions import Categorical
from torch.distributions import RelaxedOneHotCategorical
from egg.core.util import BaseEarlyStopper


class EarlyStopperAccuracy(BaseEarlyStopper):
    """
    Early stopping based on development accuracy.
    Stop if development acc. doesn't improve after threshold.
    """
    def __init__(self, threshold):
        super(EarlyStopperAccuracy, self).__init__()
        self.threshold = threshold
        self.counter = 0
        self.initial_acc = 0.

    def should_stop(self):
        validation_acc = self.validation_stats[-1][1]['question_acc']
        if validation_acc <= self.initial_acc:
            self.counter += 1
        else:
            self.counter = 0
        self.initial_acc = validation_acc
        return self.counter > self.threshold


class FactEmbedder(nn.Module):
    """
    Embed fact according to agent's role (FA or QA).
    We use three different read functions here: concat100, concat125, and concat175.
    The read function feat-value is used for sanity check experiments, where FA is given 
    (feature, value) instead of (new KB, value).
    """
    def __init__(self, opts, n_features, n_values):
        super(FactEmbedder, self).__init__()

        self.n_features = n_features
        self.n_values = n_values
        self.n_hidden = opts.hidden_dim
        self.read_function = opts.read_function
        self.vocab_size = opts.vocab_size
        self.embedding_dim = opts.embedding_dim

        if self.read_function == 'feat-value':
            self.fc = nn.Linear(self.n_features + self.n_values, self.n_hidden, bias=False)

        elif self.read_function == 'concat100':
            self.fc1 = nn.Linear(2 * self.n_features, self.n_hidden, bias=False)
            self.fc2 = nn.Linear(self.n_hidden, self.n_hidden, bias=False)

        elif self.read_function == "concat125":
            self.feature_embedding = core.RelaxedEmbedding(self.n_features, self.embedding_dim)
            self.fc_kb = nn.Linear(self.n_features, self.embedding_dim * self.n_features, bias=False)
            self.fc = nn.Linear(self.embedding_dim * (self.n_features + 1), self.n_hidden, bias=False)

        elif self.read_function == "concat175":
            self.feature_embedding = core.RelaxedEmbedding(self.n_features, self.embedding_dim)
            self.kb_embedding = nn.Embedding(self.n_values, self.embedding_dim)
            self.fc = nn.Linear(self.embedding_dim * (self.n_features + 1), self.n_hidden, bias=False)
            self.fc_qa = nn.Linear(self.embedding_dim * self.n_features, self.n_hidden, bias=False)

        else:
            raise ValueError("Invalid read function, available options: 'concat100', 'concat125', 'concat175'.")


    def update_kb(self, kb, onehot_feature, value):
        """
        Return a new KB, given old KB, fact feature, and fact value.
        """
        ones = torch.ones_like(onehot_feature)
        new_kb =  torch.mul((ones - onehot_feature), kb.float()) + torch.mul(onehot_feature, value.float())

        return new_kb


    def forward(self, _input, fAgent=False):
        # unpack input
        kb, feature, value, question = _input

        # create one-hot vector of the feature
        onehot_value = F.one_hot(value.view(-1), self.n_values).float()
        
        # if qAgent, we give the old KB and dummy feature slot (all zeros)
        if not fAgent:
            new_kb = kb.float()
            # we give dummy feature instead of the real question
            onehot_feature = F.one_hot(question.view(-1), self.n_features).float()
            onehot_feature = torch.zeros_like(onehot_feature)
        else:
            # if fAgent, we update the KB explicitly
            onehot_feature = F.one_hot(feature.view(-1), self.n_features).float()
            new_kb = self.update_kb(kb, onehot_feature, value)
            
        if self.read_function == "feat-value":
            # send concatenation of one-hot feature and one-hot value
            _input = torch.cat((onehot_feature, onehot_value), 1)
            _input = self.fc(_input)
            _input = torch.tanh(_input)

        elif self.read_function == 'concat100':
            _input = torch.cat((new_kb, onehot_feature), 1)
            _input = self.fc1(_input)
            _input = torch.tanh(_input)
            _input = self.fc2(_input)
            _input = torch.tanh(_input)

        elif self.read_function == 'concat125':
            feat_emb = self.feature_embedding(onehot_feature)
            cont_kb = self.fc_kb(new_kb)
            _input = torch.cat((feat_emb, cont_kb), 1)
            _input = self.fc(_input)
            _input = torch.tanh(_input)

        elif self.read_function == 'concat175':
            # out: batch_size x emb_dim
            feat_emb = self.feature_embedding(onehot_feature)
            # out: batch_size x n_features x emb_dim
            kb_emb = self.kb_embedding(new_kb.long())
            # out: batch_size x (n_features * emb_dim)
            kb_emb = torch.reshape(kb_emb, (-1, self.n_features * self.embedding_dim))
            # out: batch_size x (n_features + 1) * emb_dim
            _input = torch.cat((feat_emb, kb_emb), 1)
            _input = self.fc(_input)
            _input = torch.tanh(_input)

        return _input


class QuestionEmbedder(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super(QuestionEmbedder, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = core.RelaxedEmbedding(n_features, self.embedding_dim)

    def forward(self, question, qAgent=False):
        device = question.device
        if qAgent:
            return self.embedding(question)
        else:
            return torch.zeros([question.size()[0], self.embedding_dim], dtype=torch.float).to(device)


class ReceiverModule(nn.Module):
    def __init__(self, n_hidden, vocab_size):
        super(ReceiverModule, self).__init__()
    
        self.embedding = core.RelaxedEmbedding(vocab_size, n_hidden)

    def forward(self, message):
        return self.embedding(message)


class SenderModule(nn.Module):
    def __init__(self, n_hidden, embedding_dim, action_dim, vocab_size):
        super(SenderModule, self).__init__()

        input_dim = 2 * n_hidden + embedding_dim + action_dim
        self.output = nn.Linear(input_dim, vocab_size, bias=False)


    def forward(self, state):
        out = self.output(state)
        logits = F.log_softmax(out, dim=1)

        return logits


class AnswerModule(nn.Module):
    """
    Module for answering question.
    NOTE: prediction for fact is not used at the moment, 
    we might use it in the future when we need to update
    QA's KB.
    """

    def __init__(self, n_hidden, embedding_dim, action_dim, n_features, n_values):
        super(AnswerModule, self).__init__()
    
        input_dim = 2 * n_hidden + embedding_dim + action_dim
        self.fc = nn.Linear(input_dim, n_hidden)

        self.fact_feat_fc = nn.Linear(n_hidden, n_features)
        self.fact_value_fc = nn.Linear(n_hidden, n_values)

        self.question_feat_fc = nn.Linear(n_hidden, n_features, bias=False)
        self.question_value_fc = nn.Linear(n_hidden, n_values, bias=False)


    def forward(self, state):

        state = self.fc(state)
        state = torch.tanh(state)

        fact_feat = self.fact_feat_fc(state)
        fact_value = self.fact_value_fc(state)

        question_feat = self.question_feat_fc(state)
        question_value = self.question_value_fc(state)

        fact_feat_logits = F.log_softmax(fact_feat, dim=1)
        fact_value_logits = F.log_softmax(fact_value, dim=1)

        question_feat_logits = F.log_softmax(question_feat, dim=1)
        question_value_logits = F.log_softmax(question_value, dim=1)

        return {'question_feat': question_feat_logits,
                'question_value': question_value_logits,
                'fact_feat': fact_feat_logits,
                'fact_value': fact_value_logits
        }


class Agent(nn.Module):
    def __init__(self, sender, receiver, n_features, n_values, opts):
        super(Agent, self).__init__()

        self.sender = sender
        self.receiver = receiver
        self.n_features = n_features
        self.n_values = n_values
        self.hidden_dim = opts.hidden_dim
        self.embedding_dim = opts.embedding_dim
        self.action_dim = opts.action_dim

        self.fact_embedder = FactEmbedder(opts, n_features, n_values)
        self.question_embedder = QuestionEmbedder(n_features, opts.embedding_dim)
        self.action_embedder = core.RelaxedEmbedding(2, self.action_dim)

        self.fc_input_dim = 2 * opts.hidden_dim + opts.embedding_dim + self.action_dim
        self.fc = nn.Linear(self.fc_input_dim, opts.hidden_dim, bias=False)


    def forward(self, message, action_flag, agent_input, fAgent=False, qAgent=False):
        _, feature, _, question = agent_input

        # 1. embed message, out: batch_size x hidden_dim
        message = self.receiver(message)
        # 2. embed fact input, out: batch_size x hidden_dim
        fact = self.fact_embedder(agent_input, fAgent)
        # 3. embed question, which is the feature slot, out: batch_size x emb_dim
        question = self.question_embedder(question.squeeze(1), qAgent)
        # 4. encode QA flag, out: batch_size x 10
        action = self.action_embedder(action_flag)

        state = torch.cat((message, fact, question, action), dim=1)
        message_logits = self.sender(state)

        return message_logits, state


class ReinforceWrapper(nn.Module):
    def __init__(self, agent):
        super(ReinforceWrapper, self).__init__()
        self.agent = agent
        self.n_features = agent.n_features
        self.n_values = agent.n_values
        self.action_dim = agent.action_dim
        self.hidden_dim = agent.hidden_dim
        self.embedding_dim = agent.embedding_dim

    def forward(self, *args, **kwargs):
        message_logits, answer_logits = self.agent(*args, **kwargs)

        # sample message
        distr = Categorical(logits=message_logits)
        entropy = distr.entropy()

        if self.training:
            message = distr.sample()
        else:
            message = message_logits.argmax(dim=1)

        log_prob = distr.log_prob(message)

        return message, answer_logits, log_prob, entropy



class ReinforceDeterministicWrapper(nn.Module):
    def __init__(self, agent):
        super(ReinforceDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)

        return out, torch.zeros(1).to(out['feat'].device), torch.zeros(1).to(out['feat'].device)



# GAME
class SymbolGameReinforce(nn.Module):
    """
    A single-symbol Sender/Receiver game implemented with Reinforce.
    """
    def __init__(self, agentA, agentB, loss, agent_type, fa_entropy_coeff=0.0, qa_entropy_coeff=0.0):
        super(SymbolGameReinforce, self).__init__()
        self.agentA = agentA
        self.agentB = agentB
        self.loss = loss
        self.agent_type = agent_type

        self.answerModA = AnswerModule(agentA.hidden_dim, agentA.embedding_dim, agentA.action_dim, agentA.n_features, agentA.n_values)
        self.answerModB = AnswerModule(agentB.hidden_dim, agentB.embedding_dim, agentB.action_dim, agentB.n_features, agentB.n_values)

        self.fa_entropy_coeff = fa_entropy_coeff
        self.qa_entropy_coeff = qa_entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, game_input, labels):
        device = game_input[0].device

        if self.agent_type == 'symmetric':
            agentA_start = bool(np.random.binomial(1, 0.5, size=1))
        else:
            agentA_start = True
        if agentA_start:
            fAgent, qAgent = self.agentA, self.agentB
            fAnswer, qAnswer = self.answerModA, self.answerModB
        else:
            fAgent, qAgent = self.agentB, self.agentA
            fAnswer, qAnswer = self.answerModB, self.answerModA

        fa_log_probs, fa_entropies = 0., 0.
        qa_log_probs, qa_entropies = 0., 0.

        # start with a dummy message, index 0
        batch_size = labels.size(0)
        message = torch.zeros([batch_size], dtype=torch.int64).to(device)

        # action flag: 0 is for fact agent and 1 is for question agent
        fa_flag = torch.zeros([batch_size], dtype=torch.int64).to(device)
        qa_flag = torch.ones([batch_size], dtype=torch.int64).to(device)

        message, _, fa_log_prob, fa_entropy = fAgent(message, fa_flag, game_input, fAgent=True, qAgent=False)
        message, state, qa_log_prob, qa_entropy = qAgent(message, qa_flag, game_input, fAgent=False, qAgent=True)
        
        # sum entropy and log_probs
        fa_entropies += fa_entropy
        fa_log_probs += fa_log_prob

        answer = qAnswer(state)

        # for one-step game, QA log_prob and entropy does not affect anything
        qa_entropies = torch.zeros(1).to(device)
        qa_log_probs = torch.zeros(1).to(device)

        loss, rest_info = self.loss(game_input, None, None, answer, labels)
        policy_loss = ((loss.detach() - self.mean_baseline) * (fa_log_probs + qa_log_probs)).mean()
        entropy_loss = -(fa_entropies.mean() * self.fa_entropy_coeff + qa_entropies.mean() * self.qa_entropy_coeff)

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        rest_info['fa_entropy'] = fa_entropies.mean()
        rest_info['qa_entropy'] = qa_entropies.mean()

        return full_loss, rest_info
