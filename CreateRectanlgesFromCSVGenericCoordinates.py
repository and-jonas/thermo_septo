from osgeo import ogr
import os
import pandas as pd
import numpy as np

# df = pd.read_csv('Z:/Public/Jonas/003_ESWW/Design_20211001/design_for_analysis.csv')
# df = pd.read_csv('Z:/Public/Jonas/007_Thermo/ExperimentalDesign/02_Design_complete_sorted.csv')
# df = pd.read_csv('Z:/Public/Jonas/014_ESWW010/Design/01_Design_complete_sorted.csv')
# df = pd.read_csv('O:/Evaluation/Projects/KP0023_legumes/Pea/2024/design/design_sorted.csv')
df = pd.read_csv("Z:/Public/Jonas/003_ESWW/Design_20211001/design_sorted.csv")
df = df.reset_index()  # make sure indexes pair with number of rows

# File name
# outGeoJSONfn = 'Z:/Public/Jonas/007_Thermo/PyCode/ESWW006_test.geojson'
outGeoJSONfn = 'Z:/Public/Jonas/003_ESWW/Design_20211001/ESWW006_geo.geojson'
# outGeoJSONfn = '[50:]ESWW010_geo.geojson'
# outGeoJSONfn = 'O:/Evaluation/Projects/KP0023_legumes/Pea/2024/design/FPSE00x_geo.geojson'

# Input Data
width_rectan = 8
hight_rectan = 11
spacing_width = 7
spacing_hight = 9

base_shape = np.array([
    [0, 0],
    [width_rectan, 0],
    [width_rectan, hight_rectan],
    [0, hight_rectan],
    [0, 0]])

# create fields
idField = ogr.FieldDefn('plot_UID', ogr.OFTString)
genField = ogr.FieldDefn('genotype_name', ogr.OFTString)
rowField = ogr.FieldDefn('row_lot', ogr.OFTInteger)
rangeField = ogr.FieldDefn('range_lot', ogr.OFTInteger)
treatmentField = ogr.FieldDefn('treatment', ogr.OFTString)

# Create the output shapefile
GeoJSONDriver = ogr.GetDriverByName("GeoJSON")
if os.path.exists(outGeoJSONfn):
    GeoJSONDriver.DeleteDataSource(outGeoJSONfn)
outDataSource = GeoJSONDriver.CreateDataSource(outGeoJSONfn)
outLayer = outDataSource.CreateLayer(outGeoJSONfn, geom_type=ogr.wkbPoint)

outLayer.CreateField(idField)
outLayer.CreateField(genField)
outLayer.CreateField(rowField)
outLayer.CreateField(rangeField)
outLayer.CreateField(treatmentField)

for index, row in df.iterrows():
    print(row['plot_UID'], row['genotype_name'], row['row_lot'], row['range_lot'], row['treatment_name'])

    x_internal = [((width_rectan + spacing_width) * (int(row['row_lot']))) + cord[0] for cord in base_shape]
    y_internal = [((hight_rectan + spacing_hight) * (int(row['range_lot']))) + cord[1] for cord in base_shape]
    print(x_internal)

    # Create ring
    square = ogr.Geometry(ogr.wkbLinearRing)
    square.AddPoint(float(x_internal[0]), float(y_internal[0]))
    square.AddPoint(float(x_internal[1]), float(y_internal[1]))
    square.AddPoint(float(x_internal[2]), float(y_internal[2]))
    square.AddPoint(float(x_internal[3]), float(y_internal[3]))
    square.AddPoint(float(x_internal[4]), float(y_internal[4]))

    # Create polygon
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(square)
    print(polygon.ExportToWkt())

    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(polygon)
    outFeature.SetField('plot_UID', row['plot_UID'])
    outFeature.SetField('genotype_name', row['genotype_name'])
    outFeature.SetField('row_lot', row['row_lot'])
    outFeature.SetField('range_lot', row['range_lot'])
    outFeature.SetField('treatment', row['treatment_name'])

    outLayer.CreateFeature(outFeature)

    # dereference the feature
    outFeature = None

# Save and close DataSources
outDataSource = None

