cfg = {
	'MIN_NUM': 3,
	'MAX_NUM': 10,
	'ZMIN': -4.0, 	# Vertical
	'ZMAX': 4.0,
	'XMIN': -6.0,
	'XMAX': 6.0,
	'YMIN': -1.0,
	'YMAX': 11.0,
	'H': 128,
	'W': 128,
	'PAD': 20,
	'CLASSES': ("bathtub", "bed", "bookshelf", "box", "chair", 
				"counter", "desk", "door", "dresser", "garbage_bin", 
 				"lamp", "monitor", "night_stand", "pillow", "sink", 
				 "sofa", "table", "tv", "toilet"),
	# Trainer configs
	'data_root': 'dataset/',
	'cache_dir': 'cache/',
	'data_path': 'dataset/SUNRGBDMeta3DBB_v2.mat',
	'epochs': 20,
	'lr': 0.001,
	'momentum': 0.9,
	'lr_decay': 0.1,
	'batch_size': 32,
	'num_workers': 4,
	'optimizer': 'Adam', # 'Adam' or 'SGD'
	'use_cuda': True,
	'hinge_loss_margin': 0.5,
	'log_every': 100,
	'val_every': 100
}