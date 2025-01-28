
import numpy as np
import json

# path = 'Z:/Public/Jonas/007_Thermo/PyCode/ESWW006_test.geojson'
# output_path = 'Z:/Public/Jonas/007_Thermo/PyCode/ESWW006_test_new.geojson'
# path = 'Z:/Public/Jonas/014_ESWW010/Design/ESWW010_geo.geojson'
# path = 'Z:/Public/Jonas/007_Thermo/ExperimentalDesign/ESWW008_geo.geojson'
# output_path = 'Z:/Public/Jonas/007_Thermo/ExperimentalDesign/ESWW008_geo_spc.geojson'
# path = 'Z:/Public/Jonas/003_ESWW/Design_20211001/ESWW006_geo.geojson'
# output_path = 'Z:/Public/Jonas/003_ESWW/Design_20211001/ESWW006_geo_spc.geojson'
# path = 'Z:/Public/Jonas/003_ESWW/Design_20211001/ESWW006_geo.geojson'
# output_path = 'Z:/Public/Jonas/003_ESWW/Design_20211001/ESWW006_geo_spc.geojson'
# path = 'O:/Evaluation/Projects/KP0023_legumes/Pea/2024/design/FPSE00x_geo.geojson'
# output_path = 'O:/Evaluation/Projects/KP0023_legumes/Pea/2024/design/FPSE00x_spc_geo_new.geojson'
path = 'O:/Evaluation/Projects/KP0023_legumes/Pea/2024/design/FPSE00x_geo.geojson'
output_path = 'O:/Evaluation/Projects/KP0023_legumes/Pea/2024/design/FPSE00x_spc_geo_1.geojson'

# # JONAS ESWW008 SPC
# input_bottom_left = [15, 420]  # inverse coordinate system !! (QGIS: (y, x); here: (x, y))
# input_bottom_right = [15, 491]
# input_top_right = [248, 491]
# input_top_left = [248, 420]
# # JONAS ESWW010 FLIR
# input_bottom_left = [188, 20]  # inverse coordinate system !! (QGIS: (y, x); here: (x, y))
# input_bottom_right = [15, 20]
# input_top_right = [15, 411]
# input_top_left = [188, 411]
# # BEAT PEAS FLIR
# input_bottom_left = [233, 60]  # inverse coordinate system !! (QGIS: (y, x); here: (x, y))
# input_bottom_right = [60, 60]
# input_top_right = [30, 371]
# input_top_left = [353, 371]
# # JONAS ESWW010 SPC
# input_bottom_left = [173, 40]  # inverse coordinate system !! (QGIS: (y, x); here: (x, y))
# input_bottom_right = [30, 60]
# input_top_right = [45, 390]
# input_top_left = [173, 390]
# # JONAS ESWW006 FLIR
# input_bottom_left = [15, 20]  # inverse coordinate system !! (QGIS: (y, x); here: (x, y))
# input_bottom_right = [15, 191]
# input_top_right = [277, 191]
# input_top_left = [308, 20]
# beat FSP 2024
input_bottom_left = [30, 370]  # inverse coordinate system !! (QGIS: (y, x); here: (x, y))
input_bottom_right = [353, 370]
input_top_right = [248, 60]
input_top_left = [30, 60]

# # JONAS ESWW008 FLIR
# output_bottom_left = [-243, -100]
# output_bottom_right = [-606, 257]
# output_top_right = [-109, 731]
# output_top_left = [38, 483]
# # JONAS ESWW008 SPC
# output_bottom_left = [400, -736]
# output_bottom_right = [630, -745]
# output_top_right = [704, -140]
# output_top_left = [336, -118]
# # JONAS ESWW010 FLIR
# output_bottom_left = [-83, -232]
# output_bottom_right = [253, -588]
# output_top_right = [726, -99]
# output_top_left = [490, 41]
# # BEAT PEAS FLIR
# output_bottom_left = [5, -353]
# output_bottom_right = [388, -475]
# output_top_right = [617, -40]
# output_top_left = [161, -5]
# # JONAS ESWW010 SPC
# output_bottom_left = [1275, -465]
# output_bottom_right = [670, -37]
# output_top_right = [24, -751]
# output_top_left = [350, -940]
# # JONAS ESWW006 FLIR
# output_bottom_left = [693, -305]
# output_bottom_right = [584, -126]
# output_top_right = [44, -138]
# output_top_left = [34, -519]
# # JONAS ESWW006 SPC
# output_bottom_left = [-42, -359]
# output_bottom_right = [159, -683]
# output_top_right = [1144, -672]
# output_top_left = [1155, 35]
# BEAT FSP 2024 SPC
output_bottom_left = [85, -839]
output_bottom_right = [923, -914]
output_top_right = [1262, -295]
output_top_left = [363, 5]

def get_homography_matrix(source, destination):
    """ Calculates the entries of the Homography matrix between two sets of matching points.
    Args
    ----
        - `source`: Source points where each point is int (x, y) format.
        - `destination`: Destination points where each point is int (x, y) format.
    Returns
    ----
        - A numpy array of shape (3, 3) representing the Homography matrix.
    Raises
    ----
        - `source` and `destination` is lew than four points.
        - `source` and `destination` is of different size.
    """
    assert len(source) >= 4, "must provide more than 4 source points"
    assert len(destination) >= 4, "must provide more than 4 destination points"
    assert len(source) == len(destination), "source and destination must be of equal length"
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))


if __name__ == "__main__":
    source_points = np.array([
        input_bottom_left,
        input_bottom_right,
        input_top_right,
        input_top_left
    ])

    destination_points = np.array([
        output_bottom_left,
        output_bottom_right,
        output_top_right,
        output_top_left
    ])

    h = get_homography_matrix(source_points, destination_points)


h = get_homography_matrix(source_points, destination_points)


with open(path) as f:
    data = json.load(f)

x_coords = []
y_coords = []
for feature in data['features']:
    print(feature['geometry']['coordinates'])
    for point in feature['geometry']['coordinates'][0]:
        x_coords.append(point[0])
        y_coords.append(point[1])
        point_coords = [point[0], point[1], 1]
        point_coords_transformed = np.dot(h, point_coords)
        point_coords_transformed = point_coords_transformed / point_coords_transformed[2]
        point[0] = point_coords_transformed[0]
        point[1] = point_coords_transformed[1]

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)