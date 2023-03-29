
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: 007_Thermo
# Use: Extract zonal statistics from thermal images captured with stationary camera
# Last edited. 2023-03-29
# ======================================================================================================================


import matplotlib
matplotlib.use('Qt5Agg')
import cv2
from matplotlib import path
import numpy as np
import pandas as pd
import copy
import geojson
from pathlib import Path
import glob
import os
from PIL import Image
import multiprocessing
from multiprocessing import Manager, Process


class TemperatureExtractor:

    def __init__(self, dirs_to_process, dir_output, img_type, overwrite):
        self.dirs_to_process = Path(dirs_to_process),
        self.path_output = Path(dir_output)
        self.path_checker = self.path_output / "Checker"
        self.path_data = self.path_output / "Data"
        self.img_type = img_type
        self.overwrite = overwrite

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
        files = []
        for d in self.dirs_to_process:
            files.extend(glob.glob(f'{d}/*.{self.img_type}'))
        return files

    def process_images_seq(self):

        self.prepare_workspace()
        files = self.file_feed()

        for file in files:

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.img_type, ".png")
            csv_name = base_name.replace("." + self.img_type, ".csv")

            img = Image.open(file)
            img_array = np.array(img)

            # output paths
            output_name = self.path_data / csv_name

            if not self.overwrite and os.path.exists(output_name):
                continue

            # create a copy of the image
            im = copy.copy(img_array)

            with open("Z:/Public/Jonas/007_Thermo/ESWW006_plot_shapes_w_id.geojson", 'r') as infile:
                polygon_mask = geojson.load(infile)

            polygons = polygon_mask["features"]

            # data container
            data = []

            # iterate over polygons
            for polygon in polygons:

                coordinates = polygon["geometry"]["coordinates"][0][0]
                id = polygon["properties"]["id"]
                plot_label = "ESWW0060" + str(id).zfill(3)

                # iterate over corners
                vertices = []
                for c in coordinates:
                    c = [int(abs(a)) for a in c]
                    cv2.circle(im, (c[0], c[1]), 2, 12, -1)
                    vertices.append(c)
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
                # values_percentiles = np.percentile(zoneraster, np.arange(0, 101, 1))

                # assemble output
                value_stat = {'plot_UID': plot_label, 'count': count,
                              'mean': mean, 'median': median,
                              'sd': sd, 'var': var}

                data.append({**value_stat})

            df = pd.DataFrame(data, columns=data[0].keys())
            df.to_csv(self.path_data / csv_name, index=False)

    def process_image(self, work_queue, result):

        for job in iter(work_queue.get, 'STOP'):

            file = job['file']

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.img_type, ".png")
            csv_name = base_name.replace("." + self.img_type, ".csv")

            img = Image.open(file)
            img_array = np.array(img)

            # output paths
            output_name = self.path_data / csv_name

            if not self.overwrite and os.path.exists(output_name):
                continue

            # create a copy of the image
            im = copy.copy(img_array)

            with open("Z:/Public/Jonas/007_Thermo/ESWW006_plot_shapes_w_id.geojson", 'r') as infile:
                polygon_mask = geojson.load(infile)

            polygons = polygon_mask["features"]

            # data container
            data = []

            # iterate over polygons
            for polygon in polygons:

                coordinates = polygon["geometry"]["coordinates"][0][0]
                id = polygon["properties"]["id"]
                plot_label = "ESWW0060" + str(id).zfill(3)

                # iterate over corners
                vertices = []
                for c in coordinates:
                    c = [int(abs(a)) for a in c]
                    cv2.circle(im, (c[0], c[1]), 2, 12, -1)
                    vertices.append(c)
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
                # values_percentiles = np.percentile(zoneraster, np.arange(0, 101, 1))

                # assemble output
                value_stat = {'plot_UID': plot_label, 'count': count,
                              'mean': mean, 'median': median,
                              'sd': sd, 'var': var}
                data.append({**value_stat})

            df = pd.DataFrame(data, columns=data[0].keys())
            df.to_csv(self.path_data / csv_name, index=False)
            result.put(file)

    def process_images_par(self):

        self.prepare_workspace()
        files = self.file_feed()

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
                print(file, "to queue")
                job = dict()
                job['file'] = file
                jobs.put(job)

            # Start processes
            for w in range(multiprocessing.cpu_count() - 2):
                p = Process(target=self.process_image,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(files)) + " jobs started, " + str(multiprocessing.cpu_count() - 2) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()
