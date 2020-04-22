cfg = {
	'MIN_NUM': 5,
	'MAX_NUM': 15,
	'ZMIN': -4.0, 	# Vertical
	'ZMAX': 4.0,
	'XMIN': -6.0,
	'XMAX': 6.0,
	'YMIN': -1.0,
	'YMAX': 11.0,
	'H': 128,
	'W': 128,
	'PAD': 20,
	# Trainer configs
	'data_root': 'data/',
	'cache_dir': 'cache/',
	'data_path': 'data/SUNRGBDMeta3DBB_v2.mat',
	'epochs': 20,
	'lr': 0.001,
	'momentum': 0.9,
	'lr_decay': 0.1,
	'batch_size': 32,
	'num_workers': 4,
	'optimizer': 'Adam', # 'Adam' or 'SGD'
	'use_cuda': True,
	'hinge_loss_margin': 0.5
}