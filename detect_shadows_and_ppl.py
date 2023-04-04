
from PIL import Image
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import geojson
from matplotlib import path
import cv2
import scipy.ndimage
import copy
import utils
import imageio
import glob
import os

source_dir = 'O:/Evaluation/Hiwi/2022_PhotoInd_Anderegg_Jonas_MSP4/FLIR_corrected_complete'
files = glob.glob(f'{source_dir}/*.tif')
workdir = 'Z:/Public/Jonas/007_Thermo'
path_geojson = f'{workdir}/ESWW006_plot_shapes_generated_adjusted.geojson'
exp_path = path.Path([[660, 286], [570, 126], [2, 140], [83, 450]])

sample_image = Image.open('O:/Evaluation/Hiwi/2022_PhotoInd_Anderegg_Jonas_MSP4/FLIR_corrected_complete/220609-230514-689_685_FLIR.tif')
img_array = np.array(sample_image)

dummy_image = np.zeros_like(img_array)

# mask everything outside the experiment
x_coords = np.arange(0, dummy_image.shape[1])
y_coords = np.arange(0, dummy_image.shape[0])
coords = np.transpose([np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))])
mask = np.zeros_like(dummy_image)
sample_mask_image = exp_path.contains_points(coords, radius=1)
sample_mask_image = np.swapaxes(sample_mask_image.reshape(dummy_image.shape[1], dummy_image.shape[0]), 0, 1)
mask = np.where(sample_mask_image, 1, mask)
img_ = np.where(mask, img_array, 0)

# extract warmest 0.15 percentile
values_percentiles = np.percentile(img_, 90)
img__ = np.where(img_ > values_percentiles, 1, 0)
plt.imshow(img__)

with open(path_geojson, 'r') as infile:
    polygon_mask = geojson.load(infile)

polygons = polygon_mask["features"]

# iterate over polygons
plot_paths = []
for polygon in polygons:
    coordinates = polygon["geometry"]["coordinates"][0]
    plot_label = polygon["properties"]["plot_UID"]
    vertices = []
    for c in coordinates:
        c = [int(abs(a)) for a in c[:2]]
        vertices.append(c)
    plot_paths.append(path.Path(vertices, closed=True))

# Create mask of plot
# Create coordinate matrix to check if image pixel in plot polygon
x_coords = np.arange(0, img_array.shape[1])
y_coords = np.arange(0, img_array.shape[0])
coords = np.transpose([np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))])

mmask = np.zeros_like(mask)
for p in plot_paths:
    sample_mask_image = p.contains_points(coords, radius=1)
    sample_mask_image = np.swapaxes(sample_mask_image.reshape(img_array.shape[1], img_array.shape[0]), 0, 1)
    mmask = np.where(sample_mask_image, 1, mmask).astype("uint8")

mask_dil = cv2.dilate(mmask, kernel = np.ones((13, 9), np.uint8))
plt.imshow(mask_dil)

mmask = np.bitwise_not(mask_dil)

soil_mask = np.bitwise_and(img__, mmask).astype("uint8")
soil_mask = soil_mask * 255
plt.imshow(soil_mask)
imageio.imwrite('Z:/Public/Jonas/007_Thermo/Meta/soil_mask.png', soil_mask)

soil_mask = Image.open('Z:/Public/Jonas/007_Thermo/Meta/soil_mask.png')
soil_mask = np.asarray(soil_mask)

for f in files:

    base_name = os.path.basename(f)
    png_name = base_name.replace(".tif", ".png")

    print(base_name)

    img = Image.open(f)
    img_array = np.array(img)

    # mask everything outside the experiment
    x_coords = np.arange(0, img_array.shape[1])
    y_coords = np.arange(0, img_array.shape[0])
    coords = np.transpose([np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))])
    mask = np.zeros_like(img_array)
    sample_mask_image = exp_path.contains_points(coords, radius=1)
    sample_mask_image = np.swapaxes(sample_mask_image.reshape(img_array.shape[1], img_array.shape[0]), 0, 1)
    mask = np.where(sample_mask_image, 1, mask)
    img_ = np.where(mask, img_array, 0)

    # mask soil pixels
    img_ = np.where(soil_mask, 0, img_)

    # extract warmest 0.15 percentile
    values_percentiles = np.percentile(img_, 99.85)
    img__ = np.where(img_ > values_percentiles, 1, 0)
    out = scipy.ndimage.morphology.binary_fill_holes(img__).astype("uint8")
    mask = cv2.medianBlur(out, 5)

    # keep only large objects
    out = utils.filter_objects_size(mask=mask, size_th=100, dir="smaller")
    check_img = copy.copy(img_array)

    # plot bounding boxes
    coords, checker = utils.get_bounding_boxes(mask=out, check_img=check_img)
    checker = ((checker - checker.min()) * (1 / (checker.max() - checker.min()) * 255)).astype('uint8')
    plt.imshow(checker)

    # save checker image
    imageio.imwrite(f'Z:/Public/Jonas/007_Thermo/Output/Checker2/{png_name}', checker)
