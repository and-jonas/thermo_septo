
from TemperatureExtractor import TemperatureExtractor

workdir = 'Z:/Public/Jonas/007_Thermo'
workdir = '/home/anjonas/public/Public/Jonas/007_Thermo'


def run():
    dirs_to_process = f'{workdir}/TestImages'
    path_geojson = f'{workdir}/ESWW006_plot_shapes_generated.geojson'
    dir_output = f'{workdir}/Output'
    temperature_extractor = TemperatureExtractor(
        dirs_to_process=dirs_to_process,
        dir_output=dir_output,
        img_type="tif",
        overwrite=True,
    )
    temperature_extractor.process_images_par()


if __name__ == "__main__":
    run()
