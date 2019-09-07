import sys
import glob, os

import png
from scipy import stats
from scipy import ndimage
from scipy.ndimage.measurements import label
import cv2
import numpy as np


homeDir= "C:/Users/Pray/"

file= sys.argv[1].replace('~', homeDir)
print(file)
EXTENSION= ".png"

# Load mask
mask = cv2.imread(f"{file}", 0)
#print(np.unique(mask)

# Fill holes
fill = ndimage.binary_fill_holes(mask).astype(np.uint8)
if np.amax(fill) < 255:
	fill = fill * 255
png.from_array(np.uint8(fill), mode="L").save(f'{file}')

structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
labeled, ncomponents = label(fill, structure) # enumerate disjoint blobs

# move blobs to separate array
arrs = np.zeros((fill.shape[0], fill.shape[1], ncomponents), dtype=np.int)
for i in range(fill.shape[0]):
	for j in range(fill.shape[1]):
		if labeled[i, j] != 0:
			arrs[i, j, labeled[i, j] - 1] = 1

if ncomponents == 0:
	ncomponents = 1
	arrs = np.zeros((fill.shape[0], fill.shape[1], ncomponents), dtype=np.int)


# write blobs to file
for i in range(ncomponents):
	fileDir= file[:-len(EXTENSION)] + "\\"
	#print(f"\t{fileDir}1-{i}.png")

	if not os.path.exists(fileDir):
		os.makedirs(fileDir)
	png.from_array(np.uint8(arrs[:,:,i])*255, mode="L").save(f'{fileDir}1-{i}.png')
	print(f'\t{fileDir}1-{i}.png')