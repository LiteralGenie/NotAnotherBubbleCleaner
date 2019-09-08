import glob, os

import png
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import cv2
import numpy as np

DATASET_DIR= os.path.abspath("../dataset_manga/")
#EXTENSION= ".png"
OUT_DIR= DATASET_DIR + "/preview/"

print(f'Dataset: {DATASET_DIR}\nOut: {OUT_DIR}')




def blend_transparent(face_img, overlay_t_img):
	# Split out the transparency mask from the colour info
	overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
	overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

	# Again calculate the inverse mask
	background_mask = 255 - overlay_mask

	# Turn the masks into three channel, so we can use them as weights
	overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
	background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

	# Create a masked out face image, and masked out overlay
	# We convert the images to floating point in range 0.0 - 1.0
	face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
	overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

	# And finally just add them together, and rescale it back to an 8bit integer image
	return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

# ==============================================================================





# Loop thru folders (train / test / val)
for dir in glob.glob(f'{DATASET_DIR}/*/'):
	# Loop thru mask files
	for imPath in glob.glob(f"{dir}images/*"):
		try:
			file= imPath.replace("images", "masks")
			print(imPath)

			# Load image
			im= cv2.imread(f"{imPath}", -1)
			if len(im.shape) == 2:
				im= np.stack((im,im,im), axis=2)
			# Load mask
			mask = cv2.imread(f"{file.replace('.jpg','.png')}", -1)
			if mask.shape[0] != im.shape[0] or mask.shape[1] != im.shape[1]:
				print("Warning:",im.shape,mask.shape)
			if len(mask.shape) == 2:
				mask= np.dstack((mask,mask,mask))
			mask[:,:,0:2]= 0
			mask[:,:,2]= np.multiply(np.divide(mask[:,:,2],255),174)
			mask= mask[:,:,0:3]
			im= im[:,:,0:3]


			# Make RGBA version of mask
			src = mask
			tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
			_, alpha = cv2.threshold(tmp, 0, int(255*.6), cv2.THRESH_BINARY)
			b, g, r = cv2.split(src)
			rgba = [b, g, r, alpha]
			dst = cv2.merge(rgba, 4)

			# We load the images
			face_img = im
			overlay_t_img = dst

			#print(file, mask.shape, dst.shape, "\n", imPath, im.shape)
			result_2 = blend_transparent(face_img, overlay_t_img)
			# cv2.imshow('image', result_2)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			# continue


			# Convert from BGR to RGB
			temp= np.copy(result_2)
			# temp[:,:,0]= result_2[:,:,2]
			# temp[:,:,2]= result_2[:,:,0]
			result_2= temp


			# write blobs to file
			filePath= os.path.join(OUT_DIR, os.path.basename(imPath))
			if not os.path.exists(OUT_DIR):
				print(f"Making {OUT_DIR}")
				os.makedirs(OUT_DIR)
			print(filePath)
			#print(result_2.shape)
			cv2.imwrite(filePath, result_2)
			#png.from_array(np.uint8(result_2), mode="RGB").save(f'{filePath}')
		except Exception as e:
			print(e.__traceback__)
			continue