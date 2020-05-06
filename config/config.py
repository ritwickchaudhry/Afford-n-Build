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
	'rotate_probability' : 0.5,
	'CLASSES': ("bathtub", "bed", "bookshelf", "box", "chair", 
				"counter", "desk", "door", "dresser", "garbage_bin", 
 				"lamp", "monitor", "night_stand", "pillow", "sink", 
				 "sofa", "table", "tv", "toilet"),
	'TIERS': [0.5, 1.0],
	# Shuffle scenes
	'max_num_place_on_top': 2,
	# Hill Climbing
	'next_probs' : [0.2, 0.4, 0.4],	# [place_on_top, teleport, rotate]
	# Trainer configs
	'model_type': 'simple_cnn',
	'data_root': 'dataset/',
	'cache_dir': 'cache/',
	'data_path': 'dataset/SUNRGBDMeta3DBB_v2.mat',
	'epochs': 50,
	'lr': 0.001,
	'momentum': 0.9,
	'lr_decay': 0.1,
	'batch_size': 32,
	'num_workers': 4,
	'optimizer': 'Adam', # 'Adam' or 'SGD'
	'use_cuda': True,
	'hinge_loss_margin': 1.5,
	'log_every': 10,
	'val_every': 50,
	'best_model_path': 'models/simple_cnn_best.pth',
	'latest_model_path': 'models/simple_cnn_ckpt.pth'
}