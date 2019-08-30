import os
import sys
import numpy as np

from collections import defaultdict



def mi(x, y):
    return round(sklearn.metrics.mutual_info_score(x, y) / np.log(2), 2)


def gather_sample_fa(fname):

	kb_input = defaultdict(int)
	fact_input = defaultdict(int)

	# symbol mappings
	sym2fact = defaultdict(lambda: defaultdict(int))
	fact2sym = defaultdict(lambda: defaultdict(int))
	kb2sym = defaultdict(lambda: defaultdict(int))

	with open(fname) as f:
		for line in f:
			items = line.strip().split(';')

			if len(items) != 10:
				continue

			kb, f_feat, f_value, message, question, gold_value, q_feat, q_value, _, _ = items

			new_kb = kb.split()
			new_kb[int(f_feat)] = f_value
			new_kb = ' '.join(new_kb)

			fact = new_kb + '|' + f_feat
			
			if q_value != gold_value:
				continue

			kb_input[new_kb] += 1
			fact_input[fact] += 1

			sym2fact[message][fact] += 1
			fact2sym[fact][message] += 1
			kb2sym[new_kb][message] += 1

	return {'kb_input': kb_input,
			'fact_input': fact_input,
			'sym2fact': sym2fact,
			'fact2sym': fact2sym,
			'kb2sym': kb2sym
	}


def gather_sample_qa(fname):

	kb_input = defaultdict(int)
	kb_question_input = defaultdict(int)
	kb_message_input = defaultdict(int)
	question_message_input = defaultdict(int)
	all_input = defaultdict(int)

	with open(fname) as f:
		for line in f:
			items = line.strip().split(';')

			if len(items) != 10:
				continue

			kb, f_feat, f_value, message, question, gold_value, q_feat, q_value, _, _ = items

			if q_value != gold_value:
				continue

			kb_input[kb] += 1
			kb_question_input[kb + '|' + question] += 1
			kb_message_input[kb + '|' + message] += 1
			question_message_input[question + '|' + message] += 1
			all_input[message + '|' + kb + '|' + question] += 1
			

	return {'kb_input': kb_input,
			'kb_question_input': kb_question_input,
			'kb_message_input': kb_message_input,
			'question_message_input': question_message_input,
			'all_input': all_input
	}


output_dir = '../egg/zoo/common_ground/outputs'
train_file = os.path.join(output_dir, '5_feats_run2.train')
dev_file = os.path.join(output_dir, '5_feats_run2.dev')

train_data_fa = gather_sample_fa(train_file)
train_data_qa = gather_sample_qa(train_file)

# checkFA, checkQA, or checkAll
mode = 'checkFA'

# fix QA, new FA
unseen_fa_input = 0
unseen_fa_input_kb = 0
seen_qa, seen_qa_true = 0, 0
seen_qa_kb, seen_qa_kb_true = 0, 0
seen_qa_kb_question, seen_qa_kb_question_true = 0, 0
seen_qa_kb_message, seen_qa_kb_message_true = 0, 0
seen_qa_question_message, seen_qa_question_message_true = 0, 0 


# fix FA, new QA
unseen_qa_input = 0
unseen_qa_input_kb = 0
unseen_qa_input_kb_message = 0
unseen_qa_input_kb_question = 0
unseen_qa_input_question_message = 0
seen_fa, seen_fa_true = 0, 0
seen_fa_kb, seen_fa_kb_true = 0, 0

# fix FA and QA
seen_fa_seen_qa = 0
seen_fa_seen_qa_true = 0

with open(dev_file) as f:
	for line in f:
		items = line.strip().split(';')
		if len(items) != 10:
			continue

		kb, f_feat, f_value, message, question, answer, q_feat, q_value, _, _ = items

		new_kb = kb.split()
		new_kb[int(f_feat)] = f_value
		new_kb = ' '.join(new_kb)

		old_value = kb.split()[int(f_feat)]

		fact = new_kb + '|' + f_feat

		qa_input = message + '|' + kb + '|' + question
		qa_input_kb_message = kb + '|' + message
		qa_input_kb_question = kb + '|' + question
		qa_input_question_message = question + '|' + message

		if mode == 'checkFA':
			if new_kb in train_data_fa['kb_input']:
				continue

			unseen_fa_input += 1

			if qa_input in train_data_qa['all_input']:
				seen_qa += 1
				if q_value == answer:
					seen_qa_true += 1

			if qa_input_kb_message in train_data_qa['kb_message_input']:
				seen_qa_kb_message += 1
				if q_value == answer:
					seen_qa_kb_message_true += 1

			if qa_input_kb_question in train_data_qa['kb_question_input']:
				seen_qa_kb_question += 1
				if q_value == answer:
					seen_qa_kb_question_true += 1

			if qa_input_question_message in train_data_qa['question_message_input']:
				seen_qa_question_message += 1
				if q_value == answer:
					seen_qa_question_message_true += 1

			if kb in train_data_qa['kb_input']:
				seen_qa_kb += 1
				if q_value == answer:
					seen_qa_kb_true += 1

		elif mode == 'checkQA':
			# filter out all seen QA
			if qa_input_question_message in train_data_qa['question_message_input']:
				continue

			unseen_qa_input += 1
			if fact in train_data_fa['fact_input']:
				seen_fa += 1
				if q_value == answer:
					seen_fa_true += 1

			if new_kb in train_data_fa['kb_input']:
				seen_fa_kb += 1
				if q_value == answer:
					seen_fa_kb_true += 1

		elif mode == 'checkAll':
			# check accuracy when both FA's input and QA's input are seen
			# note that this doesn't mean that the combination of the two are seen during training
			if fact in train_data_fa['fact_input'] and qa_input in train_data_qa['all_input']:
				seen_fa_seen_qa += 1
				if q_value == answer:
					seen_fa_seen_qa_true += 1


if mode == 'checkFA':
	seen_qa_stats = [seen_qa, seen_qa_kb, seen_qa_kb_message, seen_qa_kb_question, seen_qa_question_message]
	seen_qa_true_stats = [seen_qa_true, seen_qa_kb_true, seen_qa_kb_message_true, seen_qa_kb_question_true, seen_qa_question_message_true]

	print('Unseen FA, fix QA ---')
	print('\% in dev. data:', round(unseen_fa_input * 100. / 2000, 2))
	print('Breakdown accuracy:')
	print('\%total\tacc.')
	for freq, true_freq in zip(seen_qa_stats, seen_qa_true_stats):
		percentage = round(freq * 100. / unseen_fa_input, 2)
		acc = round(true_freq * 100. / freq, 2)
		print(percentage, '\t', acc)

elif mode == 'checkQA':
	seen_fa_stats = [seen_fa, seen_fa_kb]
	seen_fa_true_stats = [seen_fa_true, seen_fa_kb_true]	
	print('Unseen QA, fix FA ---')
	print('\% in dev. data:', round(unseen_qa_input * 100. / 2000, 2))
	print('Breakdown accuracy:')
	print('\%total\tacc.')
	for freq, true_freq in zip(seen_fa_stats, seen_fa_true_stats):
		if true_freq > 0:
			percentage = round(freq * 100. / unseen_qa_input, 2)
			acc = round(true_freq * 100. / freq, 2)
			print(percentage, '\t', acc)
		else:
			print(freq, true_freq)

elif mode == 'checkAll':
	percentage = round(seen_fa_seen_qa * 100. / 2000, 2)
	acc = round(seen_fa_seen_qa_true * 100. / seen_fa_seen_qa, 2)
	print(percentage, '\t', acc)

			

