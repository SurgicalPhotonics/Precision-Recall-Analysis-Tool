"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import csv
from enum import Enum
from functools import partial
from typing import Optional

import dask.array as da
import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Labels, Points

# https://github.com/napari/napari/blob/19f83a2195c55518f7b89146f704021017118679/napari/layers/points/_points_constants.py
# https://napari.org/stable/_modules/napari/layers/points/points.html
from napari.layers.points._points_constants import Mode
from napari.utils import progress
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label


class PointSelectionState(Enum):
    noState = 0
    addingTP = 1
    addingFP = 2
    addingFN = 3


class Threshold(Enum):
    isodata = partial(threshold_isodata)
    li = partial(threshold_li)
    otsu = partial(threshold_otsu)
    triangle = partial(threshold_triangle)
    yen = partial(threshold_yen)


class SegmentationDiffHighlight(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.highlight_btn = QPushButton("Highlight Differences")
        self.highlight_btn.clicked.connect(self._compute_differences)
        self.layer_combos = []
        self.gt_layer_combo = self.add_labels_combo_box("Ground Truth Layer")
        self.seg_layer_combo = self.add_labels_combo_box("Segmentation Layer")
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)
        self.layout().addWidget(self.highlight_btn)
        self.layout().addStretch()

    def add_labels_combo_box(self, label_text):
        combo_row = QWidget()
        combo_row.setLayout(QHBoxLayout())
        combo_row.layout().setContentsMargins(0, 0, 0, 0)
        new_combo_label = QLabel(label_text)
        combo_row.layout().addWidget(new_combo_label)
        new_layer_combo = QComboBox(self)
        new_layer_combo.addItems(
            [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Labels)
            ]
        )
        combo_row.layout().addWidget(new_layer_combo)
        self.layer_combos.append(new_layer_combo)
        self.layout().addWidget(combo_row)
        return new_layer_combo

    def _compute_differences(self):
        gt_layer = self.viewer.layers[self.gt_layer_combo.currentText()]
        seg_layer = self.viewer.layers[self.seg_layer_combo.currentText()]
        truth_foreground = da.where(gt_layer.data != 0, 1, 0)
        seg_foreground = da.where(seg_layer.data != 0, 1, 0)
        for i in range(len(truth_foreground)):
            if np.count_nonzero(truth_foreground[i]) == 0:
                seg_foreground[i] = da.zeros(truth_foreground[i].shape)
        diff = da.where(truth_foreground != seg_foreground, 1, 0)
        gt_layer.visible = False
        seg_layer.visible = False
        self.viewer.add_labels(diff, name="seg_gt_diff", color={1: "#fca503"})

    def _reset_layer_options(self, event):
        for combo in self.layer_combos:
            combo.clear()
            combo.addItems(
                [
                    layer.name
                    for layer in self.viewer.layers
                    if isinstance(layer, Labels)
                ]
            )


class PointBasedDataAnalyticsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.state = PointSelectionState.noState
        self.layer_combos = []
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)
        self.start_adding_tp_btn = QPushButton("Begin Adding True Positives")
        self.start_adding_tp_btn.clicked.connect(
            lambda: self.changeState(newState=PointSelectionState.addingTP)
        )
        self.layout().addWidget(self.start_adding_tp_btn)
        self.layout().addStretch()

    def add_points_combo_box(self, label_text):
        combo_row = QWidget()
        combo_row.setLayout(QHBoxLayout())
        combo_row.layout().setContentsMargins(0, 0, 0, 0)
        new_combo_label = QLabel(label_text)
        combo_row.layout().addWidget(new_combo_label)
        new_layer_combo = QComboBox(self)
        new_layer_combo.addItems(
            [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Points)
            ]
        )
        combo_row.layout().addWidget(new_layer_combo)
        self.layer_combos.append(new_layer_combo)
        self.layout().addWidget(combo_row)
        return new_layer_combo

    def changeState(self, newState: PointSelectionState):
        self.state = newState
        if self.state == PointSelectionState.noState:
            print("State is none")
        elif self.state == PointSelectionState.addingTP:
            self.start_adding_tp_btn.hide()
            self.addPointsLayer(layerName="True Positive", color="green")
            self.start_adding_fp_btn = QPushButton(
                "Begin Adding False Positives"
            )
            self.start_adding_fp_btn.clicked.connect(
                lambda: self.changeState(newState=PointSelectionState.addingFP)
            )
            self.layout().addWidget(self.start_adding_fp_btn)
        elif self.state == PointSelectionState.addingFP:
            self.start_adding_fp_btn.hide()
            self.addPointsLayer(layerName="False Positive", color="red")
            self.start_adding_fn_btn = QPushButton(
                "Begin Adding False Negatives"
            )
            self.start_adding_fn_btn.clicked.connect(
                lambda: self.changeState(newState=PointSelectionState.addingFN)
            )
            self.layout().addWidget(self.start_adding_fn_btn)
        elif self.state == PointSelectionState.addingFN:
            self.start_adding_fn_btn.hide()
            self.addPointsLayer(layerName="False Negative", color="yellow")
            self.peformAnalysisButton = QPushButton("Perform Statistics")
            self.peformAnalysisButton.clicked.connect(self.peformAnalysis)
            self.layout().addWidget(self.peformAnalysisButton)
        else:
            print("Unknown State")

    def getPointsLayer(self, named: str) -> Optional[Points]:
        for layer in self.viewer.layers:
            if layer.name.lower() == named and isinstance(layer, Points):
                return layer

        return None

    def peformAnalysis(self):
        print("performing analysis")
        tpCount = 0
        fpCount = 0
        fnCount = 0

        TPLayer = self.getPointsLayer(named="true positive")
        if TPLayer is not None:
            tpCount = len(TPLayer.data)

        FPLayer = self.getPointsLayer(named="false positive")
        if FPLayer is not None:
            fpCount = len(FPLayer.data)

        FNLayer = self.getPointsLayer(named="false negative")
        if FNLayer is not None:
            fnCount = len(FNLayer.data)

        precision = tpCount / (tpCount + fpCount)
        recall = tpCount / (tpCount + fnCount)
        fScore = 2 * ((precision * recall) / (precision + recall))
        accuracy = 100 * ((tpCount + fpCount) / (tpCount + fnCount))

        print("=============================")
        print(f"True Positive: {tpCount}")
        print(f"False Positive: {fpCount}")
        print(f"False Negative: {fnCount}")
        print("-   -   -   -   -   -   -   -")
        print(f"Precision: {precision}")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"F-Score: {fScore}")
        print("=============================")

        statistics = {
            "True Positive": tpCount,
            "False Positive": fpCount,
            "False Negative": fnCount,
            "Precision": precision,
            "Accuracy": accuracy,
            "Recall": recall,
            "F-Score": fScore,
        }
        # Open a file save dialog to choose the location to save the CSV file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "QFileDialog.getSaveFileName()",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if file_name:
            with open(f"{file_name}.csv", "w", newline="") as csvfile:
                # Iterate over the statistics dictionary and write the key-value pairs as rows
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Value"])
                for key, value in statistics.items():
                    writer.writerow([key, value])

    def addPointsLayer(self, layerName: str, color: str):
        self.gt_layer_combo = self.add_points_combo_box(layerName)
        new_points_layer = self.viewer.add_points(name=layerName)
        self.gt_layer_combo.setCurrentText(new_points_layer.name)
        new_points_layer.mode = Mode.ADD
        new_points_layer.face_color = color
        new_points_layer.current_face_color = color
        new_points_layer.size = 10
        new_points_layer.current_size = 10

    def _reset_layer_options(self, event):
        for combo in self.layer_combos:
            combo.clear()
            combo.addItems(
                [
                    layer.name
                    for layer in self.viewer.layers
                    if isinstance(layer, Points)
                ]
            )


@magic_factory
def segment_by_threshold(
    img_layer: "napari.layers.Image", threshold: Threshold
) -> "napari.types.LayerDataTuple":
    with progress(total=0):
        # need to use threshold.value to get the function from the enum member
        threshold_val = threshold.value(img_layer.data.compute())
        binarised_im = img_layer.data > threshold_val
        seg_labels = da.from_array(label(binarised_im))

    seg_layer = (seg_labels, {"name": f"{img_layer.name}_seg"}, "labels")

    return seg_layer
