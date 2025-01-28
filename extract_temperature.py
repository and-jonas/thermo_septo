import os
from TemperatureExtractor import TemperatureExtractor
import glob

os.getcwd()
# os.chdir('./')

# w1 = '/home/anjonas/kp-public/Evaluation/Hiwi/2023_PhotoInd_Anderegg_Jonas_MSP4_cp'
# w2 = '/home/anjonas/public/Public/Jonas/007_Thermo'
w1 = '/home/anjonas/kp-public/Evaluation/Hiwi/2024_PhotoInd_Keller_Beat_MSP4_cp'
w2 = '/home/anjonas/kp-public/Evaluation/Projects/KP0023_legumes/Pea/2024'

def run():
    dirs_to_process = w1
    # path_geojson = f'{w2}/ExperimentalDesign/ESWW008_geo_spc_3.geojson'
    path_geojson = f'{w2}/design/FPSE00x_geo_spc_red_10.geojson'
    dir_output = f'{w1}/Output/SPC'
    temperature_extractor = TemperatureExtractor(
        dirs_to_process=dirs_to_process,
        pattern='_10',
        path_geojson=path_geojson,
        dir_output=dir_output,
        img_type="tif",
        overwrite=False,
        n_cpus=1,
    )
    temperature_extractor.process_images_par()


if __name__ == "__main__":
    run()
