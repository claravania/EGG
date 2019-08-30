import os


model_dir = 'expected_loss_1layer/2019_07_29_03_01_04'

for f in os.listdir(model_dir):
	dir_path = os.path.join(model_dir, f)
	if os.path.isdir(dir_path):
		model_files = os.listdir(dir_path)
		for mfile in sorted(model_files)[:-1]:
			os.remove(os.path.join(dir_path, mfile))
			# print(mfile)