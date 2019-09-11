# Setup:

cd {Directory to download in.}

git clone https://github.com/LiteralGenie/Bubbles

cd ./Bubbles/

git clone https://github.com/LiteralGenie/MangaBubbles

jupyter-notebook

Open .../Bubbles/notebooks/train_gray.ipynb in Jupyter to train

Open .../Bubbles/notebooks/CleanBubbles.ipynb in Jupyter to test


# Cleaning Process:

1. Identify possible bubble masks
2. Apply thresholding to image (pixels < 240 become black, pixels > 240 become white)
3. Identify connected components in thresholded image.
4. Filter out small-sized components.
5. Filter out components whose centroid does not lie within the region of any bubble mask.
6. Fill holes in component areas and apply to original image.
