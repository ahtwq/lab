from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import numpy as np


class wirter_cm():
	def __init__(self, fpath):
		f = open(fpath, 'w')
		f.close()
		self.csvname = fpath
		
	def writer_in(self, conf_matrix_list, infolist=None, class_names=None):
		C_list = conf_matrix_list
		n = len(C_list[0])
		class_names = list(range(n)) if class_names == None else class_names
		csvfile = open(self.csvname, 'a+')
		writer = csv.writer(csvfile)

		if infolist is not None:
			writer.writerow(infolist)

		for C in C_list:
			mat = np.zeros((n+2,n+2))
			acc0 = np.sum(np.diag(C,0)) / np.sum(C)
			acc0 = round(acc0, 4)
			acc1 = (np.sum(np.diag(C,1)) + np.sum(np.diag(C,0)) + np.sum(np.diag(C,-1))) / np.sum(C)
			acc1 = round(acc1, 4)
			recall = np.diag(C) / np.sum(C,1)
			precision = np.diag(C) / np.sum(C,0)
			for i,item in enumerate(recall):
				recall[i] = round(item, 4)
			for i,item in enumerate(precision):
				precision[i] = round(item, 4)

			mat[0,0] = acc1
			mat[-1,-1] = acc0
			mat[0,1:] = list(range(n)) + [-1]
			mat[1:,0] = list(range(n)) + [-1]
			mat[-1,1:-1] = precision
			mat[1:-1,-1] = recall
			mat[1:-1,1:-1] = C

			data = mat.tolist()
			data[0][1:] = class_names + ['Recall']
			class_names_var = class_names + ['Precision']
			for i, item in enumerate(data):
				if i == 0:
					continue
				item[0] = class_names_var[i-1]

			writer.writerows(data)
			writer.writerows([' '])
		writer.writerows([' '])
		writer.writerows([' '])

if __name__ == '__main__':
	y_true = [2,1,0,1,2,5,4,3,5,4]
	y_pred = [2,0,0,1,2,0,3,3,5,4]
	C = confusion_matrix(y_true, y_pred)
	print(C)
	writer_cm(C, infolist=['train', '1', 'b', '2'], csvname=None, class_names=list('abcdef'))