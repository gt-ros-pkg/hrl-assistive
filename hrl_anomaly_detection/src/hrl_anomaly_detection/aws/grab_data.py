def grab_data(filePath, n_attrib, target_loc, n_inst):
	import numpy as np
	obj = open(filePath)
	target = np.ndarray(n_inst)
	data = np.ndarray((n_inst, n_attrib))
	for i, line in enumerate(obj):
		splitted = line.split(',')
		splitted[-1] = splitted[-1].split()[0]
		#datprint splitted
		for j, ind in enumerate(splitted):
			if j is not target_loc:
				if j > target_loc:
					data[i][j-1] = splitted[j]
				else:
					data[i][j] = splitted[j]
			else:
				target[i] = splitted[target_loc]
	return (data, target)

def split_for_cross_validate(data, target):
	from sklearn.cross_validation import ShuffleSplit
	cv_data = ShuffleSplit(data.shape[0])
	folded = []
	for train, test in cv_data:
		train = (data[train], target[train]) 
		test = (data[test], target[test])
		folded.append((train, test)) 
	return folded
