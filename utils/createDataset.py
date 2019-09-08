"""
Copies images+masks from NAME1 to NAME2 (within DATASET_DIR) while 'flattening' the directory structure.
The series and chapter number folders within NAME1 are merged together, using the folder names as prefixes for the new file in NAME2.
"""

import glob
import os
from shutil import copyfile

NAME1= 'train_original'
NAME2= 'train'

DATASET_DIR= "C:/Programming/Bubbles/dataset_manga/"

def mirror(maskPath, seriesName, chapNum, NAME1, NAME2):
	baseName= os.path.basename(os.path.splitext(maskPath)[0])
	newBaseName= seriesName + "_" + chapNum + "_" + baseName

	maskDst= str.replace(maskPath, NAME1, NAME2)
	maskDst= os.path.dirname(maskDst) + "/../../" + os.path.basename(maskDst).replace(baseName, newBaseName)
	maskDst= os.path.abspath(maskDst)

	rawPath= maskPath.replace('masks', 'images').replace('.png','.jpg')
	rawDst= rawPath.replace(NAME1, NAME2)
	rawDst = os.path.dirname(rawDst) + "/../../" + os.path.basename(rawDst).replace(baseName, newBaseName)
	rawDst = os.path.abspath(rawDst)

	if not os.path.exists(os.path.dirname(maskDst)):
		print('/tMaking ' + os.path.dirname(maskDst))
		os.makedirs(os.path.dirname(maskDst))
	if not os.path.exists(os.path.dirname(rawDst)):
		print('/tMaking ' + os.path.dirname(rawDst))
		os.makedirs(os.path.dirname(rawDst))

	copyfile(maskPath, maskDst)
	copyfile(rawPath, rawDst)

	for submask in glob.glob(os.path.splitext(maskPath)[0] + "/*.png"):
		submaskDst= submask.replace(NAME1, NAME2)
		submaskDst = os.path.dirname(submaskDst) + "/../../../" + os.path.basename(os.path.dirname(submaskDst)).replace(baseName,newBaseName) + "/" + os.path.basename(submaskDst)
		submaskDst= os.path.abspath(submaskDst)

		if not os.path.exists(os.path.dirname(submaskDst)):
			os.makedirs(os.path.dirname(submaskDst))

		copyfile(submask, submaskDst)

for series in glob.glob(f"{DATASET_DIR}{NAME1}/masks/*"):
	#print(series)
	for chap in glob.glob(series + "/*"):
		#print("/t"+chap)
		for im in glob.glob(chap + "/*.png"):
			print(im)
			mirror(im, os.path.basename(series), os.path.basename(chap),
			       NAME1,
			       NAME2)