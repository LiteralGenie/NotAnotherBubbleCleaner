# Table of Contents

- [Description](#description)
- [Setup instructions](#setup)
- [Samples](#Samples)
- [Training Details](#training-details)

## Description

Clean text from manga bubbles using an implementation of [Mask-RCNN](https://github.com/matterport/Mask_RCNN).

<!-- Tested on Windows 10 with an i5. A CUDA version of this (requiring a NVIDIA GPU + CUDA installation) can be found []. -->

Cleaning Process:
1. Identify possible bubble masks
2. Apply thresholding
3. Identify connected components in thresholded image
4. Filter out small-sized components
5. Filter out components whose centroid does not lie within any mask
6. Filter out components with insufficient overlap with underlying mask
7. Fill holes in remaining components
8. Shrink components (to prevent eating away at bubble outline).
9. Overlay whited-out components with original image.

Please refer to [this notebook](https://github.com/LiteralGenie/NotAnotherBubbleCleaner/blob/master/notebooks/clean_demo.ipynb) for further details.  

## Setup

1. Download [latest release]().

2. Unzip and run `dist/main/main.exe`.

3. (optional) Modify config.json and restart program to alter parameters.

- Parameters:
  - `DEBUG`: `true` to output debug images of the masks and etc. `false` to suppress debug images.
  - `MIN_BLOB_OVERLAP`: Minimum area of mask / region intersection. Regions that do not meet this threshold are ignored when whiting out the bubbles.
  - `MIN_BLOB_SIZE`: Minimum area of region. Regions that do not meet this threshold are ignored when whiting out the bubbles.
  - `MIN_DETECTION_CONFIDENCE`: Minimum confidence level between 0 and 1 for each mask outputted by the Mask-RCNN network. Masks that do not meet this threshold are ignored when selecting regions.
  - `MODEL_PATH`: Path to the weights file (model.h5).
  - `SAVE_PATH`: Path to output the cleaned images to.

## Samples

Top-left: Original image

Top-right: Cleaned image

Bottom-left: Predicted mask regions overlaid onto thresholded image

Bottom-right: Regions to be whited-out, derived from thresholded image + mask based on centroid / area of intersection

![](https://github.com/LiteralGenie/NotAnotherBubbleCleaner/blob/master/demo/debug-asuka_2_01_0123.png)
![](https://github.com/LiteralGenie/NotAnotherBubbleCleaner/blob/master/demo/debug-amaku_2_010.png)
![](https://github.com/LiteralGenie/NotAnotherBubbleCleaner/blob/master/demo/debug-caterpillar_92_0044.png)
![](https://files.catbox.moe/ta2n7g.png)


## Training Details

- Dataset:
  - 600 training images from 30 different series.
  - 50+ validation images from 5 different series.
  - All images are in grayscale.
  
- Training code: https://github.com/LiteralGenie/NotAnotherBubbleCleaner/blob/master/notebooks/train.ipynb
  - Trained using random upscaled, unpadded 1728x1728 crops passed to a ResNet-50 based architecture.
  - Trained using Tesla V100 on [Google Compute Engine](https://console.cloud.google.com/)
	 - Total training time was roughly 4 hours (10 epochs * 1000 steps each)

---

## TODO

1. Second pass after region selection to verify all 'holes' are text. (classification net maybe?)
2. Train on white text / black bubbles. (create examples artificially?)
3. Retrain model on larger dataset (50+ series?)
4. Upload tensorflow-gpu version
5. Options for output naming
6. Mirror links
6. Clean up imports
