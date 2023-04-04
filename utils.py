
import cv2
import numpy as np


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned


def get_bounding_boxes(mask, check_img):
    """
    Get bounding boxes of each maintained lesion in a full leaf image
    :param mask: Binary segmentation mask of the image to process
    :param check_img: A copy of the corresponding image
    :return: Coordinates of the bounding boxes as returned by cv2.boundingRect()
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect_coords = []
    for c in contours:
        # find bounding box of lesions
        x, y, w, h = cv2.boundingRect(c)
        # add buffer
        w = w + 10
        h = h + 10
        x = x - 5
        y = y - 5
        # boxes must not extend beyond the edges of the image
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        # draw bounding boxes for control
        cv2.rectangle(check_img, (x, y), (x + w, y + h), (int(np.max(check_img)), 0, 0), 1)
        coords = x, y, w, h
        rect_coords.append(coords)

    return rect_coords, check_img

