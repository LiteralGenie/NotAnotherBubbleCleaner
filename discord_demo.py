import json
import os

ROOT_DIR= os.path.abspath("./Mask_RCNN/")
MODEL_PATH = os.path.abspath("F:/downloads_trash/mask_rcnn_bubbles_0010.h5")
DATASET_DIR = os.path.abspath("./MangaBubbles/")

with open(os.path.abspath("./utils/bot_config.json")) as config_file:
	BOT_CONFIG= json.load(config_file)
	DISCORD_KEY= BOT_CONFIG['globals']['DISCORD_KEY']

print("Root:", ROOT_DIR)
print("Model:", MODEL_PATH)
print("Dataset:", DATASET_DIR)

MIN_BLOB_SIZE= 8000
MIN_BLOB_OVERLAP= 1000


import sys
import cv2
import numpy as np
import skimage
import copy
import random
import glob
import scipy
import PIL
import png as PNG
from pyupload.uploader import *


sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow


import matplotlib.pyplot as plt


def get_ax(rows=1, cols=1, size=10):
	"""Return a Matplotlib Axes array to be used in
	all visualizations in the notebook. Provide a
	central point to control graph sizes.

	Change the default size attribute to control the size
	of rendered images
	"""
	_, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
	return ax


def makeTransparent(mask, alpha=.6):
	src = mask
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	_, alpha = cv2.threshold(tmp, 0, int(255 * alpha), cv2.THRESH_BINARY)
	b, g, r = cv2.split(src)
	rgba = [b, g, r, alpha]
	dst = cv2.merge(rgba, 4)
	return dst


def blend_transparent(face_img, overlay_t_img):
	# Split out the transparency mask from the colour info
	overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
	overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

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


def flatten(arrs):
	# Flatten blobs into single matrix for viewing
	x = np.zeros((arrs.shape[0], arrs.shape[1]), dtype=np.int32)
	for i in range(arrs.shape[2]):
		x[:, :] += arrs[:, :, i]
	return x

# ==============================================================================


class InferenceConfig(Config):
	BACKBONE = 'resnet50'

	IMAGE_RESIZE_MODE = 'pad64'
	IMAGE_MIN_DIM = 1728
	IMAGE_MAX_DIM = 1728

	# TRAIN_ROIS_PER_IMAGE = 30

	# Blech
	NAME = "bubbles"
	NUM_CLASSES = 1 + 1  # background + 3 shapes
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	IMAGE_CHANNEL_COUNT = 1
	MEAN_PIXEL = np.array([123.7])


inference_config = InferenceConfig()
inference_config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=ROOT_DIR)
model.load_weights(MODEL_PATH, by_name=True)

imPath= "C:/Programming/Bubbles/MangaBubbles/train_original/images/asuka/1/01_0018.jpg"
imPath= random.choice(glob.glob("C:/Programming/Bubbles/MangaBubbles/train_original/images/asuka/1/*"))

import io
import os
import re
import shutil
import subprocess
import time
import urllib

import discord
import requests

client= discord.Client()
@client.event
async def on_ready():
	print('Logged in as')
	print(client.user.name)
	print(client.user.id)

@client.event
async def on_message(message):
	if '??clean' in message.content:
		message.channel.typing()
		write = False

		msg = None
		split = message.content.split()
		angle = "\"90\"" if len(split) < 2 else ("\"" + split[1] + "\"")
		print("angle", angle)
		print(message.channel.name, "channel")
		async for mege in message.channel.history(limit=30):
			if mege.attachments:
				print("attachment")
				msg = mege
				byes = io.BytesIO(await msg.attachments[0].read())
				with open('C:\\Users\\Pray\\Pictures\\test2.png', 'wb') as f:
					f.write(byes.getvalue())
				byes.close()

				write = True
				break
			else:
				print("no attachment")
				img = re.search("(i.imgur.com\/\w+.(png|jpg))", mege.content)
				disc = re.search("(cdn.discordapp.com\/attachments\/.+.(png|jpg))", mege.content)

				grp = ""
				if img:
					grp = img
					print('img')
				elif disc:
					print('disc')
					grp = disc
				print("grp", grp)

				if grp:
					print(grp.group())
					r = requests.get("http://" + grp.group(), stream=True)
					with open("C:/Users/Pray/Pictures/test2.png", 'wb') as out_file:
						shutil.copyfileobj(r.raw, out_file)
					write = True
					break
				else:
					continue

		if not write:
			return

		imPath= "C:/Users/Pray/Pictures/test2.png"

		original_image = skimage.io.imread(imPath, as_gray=True)
		if np.max(original_image) <= 1:
			original_image = original_image * 255
		if len(original_image.shape) == 2:
			original_image = original_image[..., np.newaxis]

		original_image = utils.resize_image(original_image, min_dim=inference_config.IMAGE_MIN_DIM,
		                                    max_dim=inference_config.IMAGE_MAX_DIM,
		                                    min_scale=inference_config.IMAGE_MIN_SCALE,
		                                    mode=inference_config.IMAGE_RESIZE_MODE)[0]
		disp_image = np.concatenate((original_image, original_image, original_image), axis=2)

		#get_ax().imshow(disp_image.astype(np.uint8))

		print('max0',np.max(disp_image))
		PIL.Image.fromarray(np.uint8(disp_image)).save(f"C:/Users/Pray/Pictures/0original.png")

		results = model.detect([original_image], verbose=1)
		r = copy.deepcopy(results[0])
		print(r['scores'])

		inds = []
		for i in reversed(range(len(r['scores']))):
			if r['scores'][i] > .98:
				break
			else:
				inds.append(i)

		r['masks'] = np.delete(r['masks'], inds, 2)
		r['scores'] = np.delete(r['scores'], inds, 0)
		r['class_ids'] = np.delete(r['class_ids'], inds, 0)
		r['rois'] = np.delete(r['rois'], inds, 0)

		im = visualize.display_instances(disp_image, r['rois'], r['masks'], r['class_ids'],
		                                 ['a'] * 123, r['scores'], ax=get_ax())
		print('max',np.max(im))
		PIL.Image.fromarray(im).save("C:/Users/Pray/Pictures/1masked.png")
		# Convert image to pure black and white
		__, greyImage = cv2.threshold(original_image, 240, 1, cv2.THRESH_BINARY)
		greyImage = (greyImage*255).astype(np.uint8)

		# Identify connected white blobs
		blobStats = cv2.connectedComponentsWithStats(greyImage, 8, cv2.CV_16U)

		# Identify larger blobs
		blobLabels = [x for x in range(len(blobStats[2])) if blobStats[2][x][4] > MIN_BLOB_SIZE and x > 0]

		# Move larger blobs to separate array
		blobs = np.zeros(original_image.shape)
		out = blobStats[1][..., np.newaxis]  # Labeled (flat) image

		for label in blobLabels[1:]:
			blobs = np.concatenate((blobs, out == label), axis=2)
		blobs = blobs * 255

		debug_list = [str(x) + ": " + str(blobStats[3][x].astype(np.uint16)) for x in blobLabels]
		print(f"Selected blobs:\n{chr(10).join(debug_list)}")
		#get_ax().imshow(greyImage.astype(np.uint8), cmap='Greys_r')

		PIL.Image.fromarray(np.uint8(greyImage)).save(f"C:/Users/Pray/Pictures/2threshold.png")

		# Display blobs post-size-filter for debug
		arr = flatten(blobs.astype(np.uint8))
		disp_arr = np.dstack((arr, arr, arr))
		#get_ax().imshow(disp_arr, cmap='Greys_r')

		PNG.from_array(np.uint8(arr), mode="L").save(f"C:/Users/Pray/Pictures/3thresholdbig.png")

		# Loop through the list of (large) blobs
		filteredBlobLabels = set([])
		for msk in range(r['masks'].shape[2]):
			mask = r['masks'][:, :, msk]

			# Select blobs whose centroid falls within the mask and whose overlap with that mask is sufficient
			for i in range(len(blobLabels)):
				center = blobStats[3][blobLabels[i]]
				center = center.astype(np.uint16)

				if mask[center[1]][center[0]] and np.count_nonzero(
						np.logical_and(mask, blobStats[1][:, :] == blobLabels[i])) > MIN_BLOB_OVERLAP:
					# print(f"Adding blob {blobLabels[i]} with center {center}")
					filteredBlobLabels.add(blobLabels[i])

		flat_mask = flatten(r['masks']) * 255
		#get_ax().imshow(flat_mask.astype(np.uint8), cmap='Greys_r')

		debug_list = [str(x) + ": " + str(blobStats[3][x]) for x in filteredBlobLabels]
		print(f"Selected blobs:\n{chr(10).join(debug_list)}")

		print('max',np.max(flat_mask))
		PIL.Image.fromarray(np.uint8(flat_mask)).save(f"C:/Users/Pray/Pictures/4masks.png")

		# Debug filtered blobs

		dotSize = 25
		grey = 126

		test = np.zeros([original_image.shape[0], original_image.shape[1], len(filteredBlobLabels)])

		for i, b in enumerate(filteredBlobLabels):
			test[:, :, i] = (blobStats[1][:, :] == b) * 255

			center = blobStats[3][b].astype(np.uint16)
			centerBlock = [center[0] - dotSize, center[0] + dotSize, center[1] - dotSize, center[1] + dotSize]

			centerBlock[1] = np.minimum(centerBlock[1], test.shape[1])
			centerBlock[3] = np.minimum(centerBlock[3], test.shape[0])
			centerBlock[0] = np.maximum(centerBlock[0], 0)
			centerBlock[2] = np.maximum(centerBlock[2], 0)

			test[centerBlock[2]:centerBlock[3], centerBlock[0]:centerBlock[1], i] = grey
		# get_ax().imshow(test[:,:,i].astype(np.uint8), cmap='Greys_r')
		# print(np.max(test[:,:,i]), np.mean(test[:,:,i]))

		flat = (flatten(test.astype(np.uint8)))
		# flat= flat / np.max(flat) * 255
		#get_ax().imshow(flat, cmap='Greys_r')

		debug_list = [str(x) + ": " + str(blobStats[3][x]) for x in filteredBlobLabels]
		print(f"Selected blobs:\n{chr(10).join(debug_list)}")

		print('max', np.max(flat))
		PIL.Image.fromarray(np.uint8(flat)).save(f"C:/Users/Pray/Pictures/5comps.png")

		import png

		cleaned_image = original_image
		for b in filteredBlobLabels:
			comp = blobStats[1][:, :] == b
			comp = comp * 255
			disp = np.dstack((comp, comp, comp))
			# get_ax().imshow(disp.astype(np.uint8), cmap='Greys_r')

			fill = scipy.ndimage.binary_fill_holes(comp).astype(np.uint8) * 255

			cleaned_image = np.maximum(fill[:, :, np.newaxis], cleaned_image)

		disp_image2 = np.concatenate((cleaned_image, cleaned_image, cleaned_image), axis=2)
		#get_ax().imshow(disp_image2.astype(np.uint8), cmap='Greys_r')

		print('max', np.max(cleaned_image))
		print(cleaned_image.dtype)
		print(cleaned_image.shape)
		PIL.Image.fromarray(disp_image2.astype(np.uint8)).save(f"C:/Users/Pray/Pictures/6final.png")

		# Debug Montage
		border = 40
		dims = [disp_image.shape[0], disp_image.shape[1], 3]
		montage = np.ones([dims[0] * 2 + border, dims[1] * 2 + border, 3]) * 255

		montage[0:dims[0], 0:dims[1], :] = disp_image
		montage[dims[0] + border:dims[0] * 2 + border, 0:dims[1], :] = im

		montage[0:dims[0], dims[1] + border:dims[1] * 2 + border, :] = disp_image2
		montage[dims[0] + border:dims[0] * 2 + border, dims[1] + border:dims[1] * 2 + border, :] = np.dstack(
			(flat, flat, flat))
		#get_ax().imshow(montage.astype(np.uint8))

		PIL.Image.fromarray(montage.astype(np.uint8)).save(f"C:/Users/Pray/Pictures/7montage.png")
		PIL.Image.fromarray(montage[dims[0]+border:2*dims[0]+border,:,:].astype(np.uint8)).save(f"C:/Users/Pray/Pictures/7.5montage.png")

		try:
			await message.channel.send(content="Debug:",file=discord.File('C:\\Users\\Pray\\Pictures\\7.5montage.png'))
		except discord.errors.HTTPException as e:
			await message.channel.send("Result too large to upload directly. Please wait for 3rd party uplaod...")
			link= CatboxUploader("C:/Users/Pray/Pictures/7montage.png").execute()
			await message.channel.send(link)
		try:
			await message.channel.send(file=discord.File('C:\\Users\\Pray\\Pictures\\6final.png'))
		except discord.errors.HTTPException as e:
			await message.channel.send("Result too large to upload directly. Please wait for 3rd party uplaod...")
			link= CatboxUploader("C:/Users/Pray/Pictures/6final.png").execute()
			await message.channel.send(link)
		return

if __name__ == "__main__":
    client.run(DISCORD_KEY)