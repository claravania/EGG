import os
import sys


model_ids = [
	('17193619_0', 80),
	('17193577_0', 96),
	('17193658_0', 196),
	('17193694_0', 97),
	('17193730_0', 74)
]

checkpoint_prefix = '../nest/single_symbol/2019_08_29_03_03_54'
output_prefix = './egg/zoo/common_ground/outputs/5feats_regularization/earlystop_wdecay'

for i, (mid, num_epoch) in enumerate(model_ids):

	eval_train = 'python -m egg.zoo.common_ground.game --train_data=./egg/zoo/common_ground/example_data/10k_qa_f5_c5.train' + \
				' --dump_data=./egg/zoo/common_ground/example_data/10k_qa_f5_c5.train --batch_size=512 --vocab_size=1000' + \
				' --read_function=concat175 --load_from_checkpoint=' + checkpoint_prefix + '/' + mid + '/' + \
				str(num_epoch) + '.tar' + ' --dump_output=' + output_prefix + '_' + str(i+1) + '.train' 

	eval_dev = 'python -m egg.zoo.common_ground.game --train_data=./egg/zoo/common_ground/example_data/10k_qa_f5_c5.train \
				--dump_data=./egg/zoo/common_ground/example_data/10k_qa_f5_c5.dev --batch_size=512 --vocab_size=1000 \
				--read_function=concat175 --load_from_checkpoint=' + checkpoint_prefix + '/' + mid + '/' + \
				str(num_epoch) + '.tar' + ' --dump_output=' + output_prefix + '_' + str(i+1) + '.dev'

	os.system(eval_train)
	os.system(eval_dev)
