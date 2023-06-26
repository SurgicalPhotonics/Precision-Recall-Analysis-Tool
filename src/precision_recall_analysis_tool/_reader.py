"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""
import os

import dask.array as da
import tifffile


def napari_get_reader(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None

    mask_image_path = os.path.join(path, "mask.tif")
    raw_image_path = os.path.join(path, "raw.tif")
    if not (
        os.path.isfile(mask_image_path) and os.path.isfile(raw_image_path)
    ):
        return None

    return reader_function


def reader_function(path):
    path = os.path.normpath(path)

    mask_image_path = os.path.join(path, "mask.tif")
    raw_image_path = os.path.join(path, "raw.tif")

    mask_image = da.from_array(tifffile.imread(mask_image_path))
    raw_image = da.from_array(tifffile.imread(raw_image_path))

    mask_layer_data = (mask_image, {"name": "mask"}, "labels")
    raw_layer_data = (raw_image, {"name": "raw"}, "image")

    return [raw_layer_data, mask_layer_data]
