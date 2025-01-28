
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: 007_Thermo
# Use: Extract zonal statistics from thermal images captured with stationary camera
# Last edited. 2023-03-29
# ======================================================================================================================


# import matplotlib
# from matplotlib import pyplot as plt
import PIL
import cv2
import scipy
from matplotlib import path
import numpy as np
import pandas as pd
import copy
import geojson
from pathlib import Path
import glob
import os
from PIL import Image
import imageio
import multiprocessing
from multiprocessing import Manager, Process
from datetime import datetime
# import exiftool
import json
import subprocess
# matplotlib.use('Qt5Agg')


# TODO enable processing on Monster


class TemperatureExtractor:

    def __init__(self, dirs_to_process, pattern, path_geojson, dir_output, img_type, overwrite, n_cpus):
        self.dirs_to_process = dirs_to_process,
        self.pattern = pattern
        self.path_geojson = path_geojson
        self.path_output = Path(dir_output)
        self.path_checker = self.path_output / "Checker"
        self.path_data = self.path_output / "Data"
        self.img_type = img_type
        self.overwrite = overwrite
        self.n_cpus = n_cpus

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        self.path_checker.mkdir(parents=True, exist_ok=True)
        self.path_data.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """
        # TODO check for already processed files and remove from list of files to process
        # this year has a unique file structure - capture this
        if self.dirs_to_process[0] == '/home/anjonas/kp-public/Evaluation/Hiwi/2022_PhotoInd_Anderegg_Jonas_MSP4_cp/FLIR_corrected_complete':
            files = glob.glob(f'{self.dirs_to_process[0]}/*{self.pattern}.{self.img_type}')
        else:
            dates = os.listdir(self.dirs_to_process[0])
            dirs_to_process = [os.path.join(self.dirs_to_process[0], d) for d in dates]
            dirs_to_process = dirs_to_process[:1]

            files = []
            for d in dirs_to_process:
                f = glob.glob(f'{d}/*/*/*{self.pattern}.{self.img_type}')
                files.extend(f)
                print(d + f": found {len(f)} files.")

            # remove already processed files from list
            if not self.overwrite:
                processed = glob.glob(f'{self.path_data}/*.csv')
                processed = [os.path.basename(x).replace(".csv", "") for x in processed]
                files = [f for f in files if os.path.basename(f).replace("." + self.img_type, "") not in processed]
                print(str(len(processed)) + " already processed. Processing " + str(len(files)))

        return files

    def process_images_seq(self):
        """
        Processes each thermal image sequentially
        """

        self.prepare_workspace()
        files = self.file_feed()
        checker_files = files[::100]

        for file in files:

            # read image
            base_name = os.path.basename(file)
            png_name = base_name.replace("." + self.img_type, ".png")
            csv_name = base_name.replace("." + self.img_type, ".csv")

            # get file creation time
            m_time = os.path.getmtime(file)
            dt_m = datetime.fromtimestamp(np.floor(m_time))

            # try:
            img = Image.open(file)
            # except PIL.UnidentifiedImageError:
            #     continue
            img_array = np.array(img)

            # output paths
            output_name = self.path_data / csv_name

            if not self.overwrite and os.path.exists(output_name):
                continue

            # create a copy of the image
            im = copy.copy(img_array)

            with open(self.path_geojson, 'r') as infile:
                polygon_mask = geojson.load(infile)

            polygons = polygon_mask["features"]

            # data container
            data = []

            # iterate over polygons
            for polygon in polygons:

                coordinates = polygon["geometry"]["coordinates"][0]
                plot_label = polygon["properties"]["plot_UID"]

                # iterate over corners
                vertices = []
                for c in coordinates:
                    c = [int(abs(a)) for a in c[:2]]
                    vertices.append(c)
                helper = np.array([tuple(x) for x in vertices])
                cv2.drawContours(im, [helper], 0,  int(np.max(im)), 1)
                # make a matplotlib path
                plot_path = path.Path(vertices, closed=True)

                # Create plot mask
                # Create coordinate matrix to check if image pixel is in plot polygon
                x = np.arange(0, img_array.shape[1])
                y = np.arange(0, img_array.shape[0])
                xy = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
                sample_mask_image = plot_path.contains_points(xy, radius=1)

                # Create masked array
                sample_mask_image = np.swapaxes(sample_mask_image.reshape(img_array.shape[1], img_array.shape[0]), 0, 1)
                zoneraster = np.ma.masked_array(img_array, np.logical_not(sample_mask_image))

                # get zonal stats
                count = zoneraster.count()
                mean = np.ma.mean(zoneraster)
                median = np.ma.median(zoneraster)
                sd = np.ma.std(zoneraster)
                var = np.ma.var(zoneraster)
                m_data = np.where(sample_mask_image, img_array, np.nan)
                perc3 = np.nanpercentile(m_data, 3)
                perc97 = np.nanpercentile(m_data, 97)
                int_perc = perc97 - perc3

                # assemble output
                value_stat = {'timestamp': str(dt_m),
                              'plot_UID': plot_label,
                              'count': count,
                              'mean': mean, 'median': median,
                              'sd': sd, 'var': var,
                              'percentile_3': perc3, 'percentile_97': perc97,
                              'int_percentile': int_perc}

                # TODO add measurement time point
                #  (either from file name, or from when it was saved on disk: check with NK)

                data.append({**value_stat})

            df = pd.DataFrame(data, columns=data[0].keys())
            df.to_csv(self.path_data / csv_name, index=False)

            # save some checker images
            if file in checker_files:
                out_image = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
                out_name = self.path_checker / png_name
                imageio.imwrite(out_name, out_image)

    def process_image(self, work_queue, result, checker_files):
        """
        Processes a single thermal image within a parallel workflow
        """

        for job in iter(work_queue.get, 'STOP'):

            file = job['file']

            # read image
            base_name = os.path.basename(file)
            png_name = base_name.replace("." + self.img_type, ".png")
            csv_name = base_name.replace("." + self.img_type, ".csv")
            dls_name = "_".join([base_name.replace("." + self.img_type, ""), "dls.csv"])

            # get file creation time
            m_time = os.path.getmtime(file)
            dt_m = datetime.fromtimestamp(np.floor(m_time))

            try:
                img = Image.open(file)
            except PIL.UnidentifiedImageError:
                continue
            img_array = np.array(img)

            # with exiftool.ExifToolHelper() as et:
            #     metadata = et.get_metadata(file)
            #
            # # Extracting DLS-related fields
            # dls_fields = [
            #     'XMP:Irradiance',
            #     'XMP:SpectralIrradiance',
            #     'XMP:EstimatedDirectLightVector',
            #     'XMP:SolarElevation',
            #     'XMP:SolarAzimuth'
            # ]
            #
            # dls_values = {field: metadata[0].get(field, 'Not Available') for field in dls_fields}
            # df = pd.DataFrame(dls_values, columns=dls_fields)
            # df.to_csv(self.path_data / dls_name, index=False)

            # output paths
            output_name = self.path_data / csv_name

            if not self.overwrite and os.path.exists(output_name):
                continue

            # create a copy of the image
            im = copy.copy(img_array)

            with open(self.path_geojson, 'r') as infile:
                polygon_mask = geojson.load(infile)

            polygons = polygon_mask["features"]

            # data container
            data = []

            # iterate over polygons
            for polygon in polygons:

                coordinates = polygon["geometry"]["coordinates"][0]
                plot_label = polygon["properties"]["plot_UID"]

                # iterate over corners
                vertices = []
                for c in coordinates:
                    c = [int(abs(a)) for a in c[:2]]
                    vertices.append(c)
                helper = np.array([tuple(x) for x in vertices])
                cv2.drawContours(im, [helper], 0,  int(np.max(im)), 1)
                # make a matplotlib path
                plot_path = path.Path(vertices, closed=True)

                # Create plot mask
                # Create coordinate matrix to check if image pixel is in plot polygon
                x = np.arange(0, img_array.shape[1])
                y = np.arange(0, img_array.shape[0])
                xy = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
                sample_mask_image = plot_path.contains_points(xy, radius=1)

                # Create masked array
                sample_mask_image = np.swapaxes(sample_mask_image.reshape(img_array.shape[1], img_array.shape[0]), 0, 1)
                zoneraster = np.ma.masked_array(img_array, np.logical_not(sample_mask_image))

                # get zonal stats
                count = zoneraster.count()

                # remove outliers
                m_data = np.where(sample_mask_image, img_array, np.nan)
                Q1 = np.nanpercentile(m_data, 25)
                Q3 = np.nanpercentile(m_data, 75)
                IQR = Q3 - Q1
                Lower = Q1 - 1.5 * IQR
                Upper = Q3 + 1.5 * IQR
                m_data_no_outlier = m_data[np.where((m_data > Lower) & (m_data < Upper))]
                mean = np.mean(m_data_no_outlier)
                median = np.median(m_data_no_outlier)
                sd = np.std(m_data_no_outlier)
                var = np.var(m_data_no_outlier)
                # assemble output
                value_stat = {'timestamp': str(dt_m),
                              'plot_UID': plot_label,
                              'count': count,
                              'count_no_outlier': m_data_no_outlier.shape[0],
                              'mean': mean, 'median': median,
                              'sd': sd, 'var': var,
                              'iqr': IQR}

                # TODO add measurement time point
                #  (either from file name, or from when it was saved on disk: check with NK)

                data.append({**value_stat})

            df = pd.DataFrame(data, columns=data[0].keys())
            df.to_csv(self.path_data / csv_name, index=False)

            # save some checker images
            if file in checker_files:
                out_image = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
                out_name = self.path_checker / png_name
                imageio.imwrite(out_name, out_image)

            result.put(file)

    def process_images_par(self):

        self.prepare_workspace()
        files = self.file_feed()
        checker_files = files[::100]

        if len(files) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(files)
            count = 0

            # Build up job queue
            for file in files:
                # print(file, "to queue")
                job = dict()
                job['file'] = file
                jobs.put(job)

            # Start processes
            for w in range(self.n_cpus):
                p = Process(target=self.process_image,
                            args=(jobs, results, checker_files))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(files)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()
