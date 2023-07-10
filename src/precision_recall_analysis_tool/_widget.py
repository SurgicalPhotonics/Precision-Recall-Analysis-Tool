"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import csv
from enum import Enum
from array import *
from functools import partial
from typing import Optional
import re
from skimage import draw
import dask.array as da
import napari
from magicgui import magic_factory
import matplotlib.path as mplPath
from napari.layers import Points, Image, Shapes, Labels

import numpy as np
# https://github.com/napari/napari/blob/19f83a2195c55518f7b89146f704021017118679/napari/layers/points/_points_constants.py
# https://napari.org/stable/_modules/napari/layers/points/points.html
from napari.layers.points._points_constants import Mode
from napari.layers.shapes._shapes_constants import Mode
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
import skimage.draw
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label
from skimage import data



class Threshold(Enum):
    isodata = partial(threshold_isodata)
    li = partial(threshold_li)
    otsu = partial(threshold_otsu)
    triangle = partial(threshold_triangle)
    yen = partial(threshold_yen)


class PointSelectionState(Enum):
    noState = 0
    addingTP = 1
    addingTN = 2
    addingFP = 3
    addingFN = 4


class PointType(Enum):
    TruePositive = 0
    TrueNegative = 1
    FalsePositive = 2
    FalseNegative = 3
    CounterItem = 4

    @property
    def abbreviation(self):
        if self == PointType.TruePositive:
            return "TP"
        elif self == PointType.TrueNegative:
            return "TN"
        elif self == PointType.FalsePositive:
            return "FP"
        elif self == PointType.FalseNegative:
            return "FN"
        elif self == PointType.CounterItem:
            return "#"

    @property
    def color(self):
        if self == PointType.TruePositive:
            return "green"
        elif self == PointType.TrueNegative:
            return "lightgreen"
        elif self == PointType.FalsePositive:
            return "orange"
        elif self == PointType.FalseNegative:
            return "red"
        elif self == PointType.CounterItem:
            return "purple"


class GeneralCounter(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.layer_combos = []
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)

        self.totalCountLabel = QLabel(f"Count: {0}")
        self.layout().addWidget(self.totalCountLabel)

        self.startCountingButton = QPushButton("Begin Counting")
        self.startCountingButton.clicked.connect(lambda: self.startCounting())
        self.layout().addWidget(self.startCountingButton)
        self.layout().addStretch()

    def startCounting(self):
        self.startCountingButton.hide()
        self.addPointsLayer(pointType=PointType.CounterItem)

    def getPointsLayer(self, named: str) -> Optional[Points]:
        for layer in self.viewer.layers:
            if layer.name.lower() == named and isinstance(layer, Points):
                return layer

        return None

    def getPointsCount(self, pointType: PointType) -> int:
        layer = self.getPointsLayer(pointType.name.lower())
        if layer is not None:
            return len(layer.data)

        print(f"layer for type {pointType.name} was none")
        return 0

    def addPointsLayer(self, pointType: PointType):
        new_points_layer = self.viewer.add_points(name=pointType.name)
        new_points_layer.mode = 'add'
        new_points_layer.face_color = pointType.color
        new_points_layer.current_face_color = pointType.color
        new_points_layer.symbol = "disc"
        new_points_layer.size = 10
        new_points_layer.current_size = 10

        def printCoordinates(event):
            last_point = event.source.data[
                -1
            ]  # Get the last point in the data array
            print(f"The last point added is at coordinates: {last_point}")
            totalNumberOfPoints = len(event.source.data)
            countString = f"Count: {totalNumberOfPoints}"
            print(countString)
            self.totalCountLabel.setText(countString)

        new_points_layer.events.data.connect(
            printCoordinates
        )  # Connect the callback function

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


class PointBasedDataAnalyticsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.state = PointSelectionState.noState
        self.layer_combos = []
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)

        self.topTPLabel = QLabel(f"TP: {0}")
        self.topTPLabel.setObjectName(PointType.TruePositive.name)
        self.layout().addWidget(self.topTPLabel)

        self.topFPLabel = QLabel(f"FP: {0}")
        self.topFPLabel.setObjectName(PointType.FalsePositive.name)
        self.layout().addWidget(self.topFPLabel)

        self.topTNLabel = QLabel(f"TN: {0}")
        self.topTNLabel.setObjectName(PointType.TrueNegative.name)
        self.layout().addWidget(self.topTNLabel)

        self.topFNLabel = QLabel(f"FN: {0}")
        self.topFNLabel.setObjectName(PointType.FalseNegative.name)
        self.layout().addWidget(self.topFNLabel)

        self.start_adding_tp_btn = QPushButton("Begin Adding True Positives")
        self.start_adding_tp_btn.clicked.connect(
            lambda: self.changeState(newState=PointSelectionState.addingTP)
        )
        self.layout().addWidget(self.start_adding_tp_btn)

        self.start_adding_tn_btn = QPushButton("Begin Adding True Negatives")
        self.start_adding_tn_btn.clicked.connect(
            lambda: self.changeState(newState=PointSelectionState.addingTN)
        )
        self.layout().addWidget(self.start_adding_tn_btn)
        self.start_adding_tn_btn.hide()

        self.start_adding_fp_btn = QPushButton("Begin Adding False Positives")
        self.start_adding_fp_btn.clicked.connect(
            lambda: self.changeState(newState=PointSelectionState.addingFP)
        )
        self.layout().addWidget(self.start_adding_fp_btn)
        self.start_adding_fp_btn.hide()

        self.start_adding_fn_btn = QPushButton("Begin Adding False Negatives")
        self.start_adding_fn_btn.clicked.connect(
            lambda: self.changeState(newState=PointSelectionState.addingFN)
        )
        self.layout().addWidget(self.start_adding_fn_btn)
        self.start_adding_fn_btn.hide()

        self.peformAnalysisButton = QPushButton("Perform Statistics")
        self.peformAnalysisButton.clicked.connect(self.peformAnalysis)
        self.layout().addWidget(self.peformAnalysisButton)
        self.peformAnalysisButton.hide()
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
            self.start_adding_tn_btn.show()
            self.addPointsLayer(pointType=PointType.TruePositive)

        elif self.state == PointSelectionState.addingTN:
            self.start_adding_tn_btn.hide()
            self.start_adding_fp_btn.show()
            self.addPointsLayer(pointType=PointType.TrueNegative)

        elif self.state == PointSelectionState.addingFP:
            self.start_adding_fp_btn.hide()
            self.start_adding_fn_btn.show()
            self.addPointsLayer(pointType=PointType.FalsePositive)

        elif self.state == PointSelectionState.addingFN:
            self.start_adding_fn_btn.hide()
            self.peformAnalysisButton.show()
            self.addPointsLayer(pointType=PointType.FalseNegative)

        else:
            print("Unknown State")

    def getPointsLayer(self, named: str) -> Optional[Points]:
        for layer in self.viewer.layers:
            if layer.name.lower() == named and isinstance(layer, Points):
                return layer

        return None

    def getPointsCount(self, pointType: PointType, polygon) -> int:

        layer = self.getPointsLayer(pointType.name.lower())
        if layer is not None:
            if polygon is None:
                return len(layer.data)
            else:
                # Create a Path object from the polygon points
                poly_path = mplPath.Path(polygon)
                return np.sum(poly_path.contains_points(layer.data))
            
        else:
            print(f"layer for type {pointType.name} was none")
            return 0

        
            

    def getAllSubsectionPolygons(self):
        layers = []
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes) and re.search(r'section\d+_polygon', layer.name):
                layers.append(layer)
                print(f"Found Subsection - {layer.name}")
        return layers
    
    def peformAnalysis(self):
        print("performing analysis")
        
        subsectionLayers = self.getAllSubsectionPolygons()

        allStatsDicts = []
        
        for layer in subsectionLayers:
            polygons = layer.data
            if len(polygons) > 0:
                print(f"{layer.name} had {len(polygons)} polygons")
                polygonToAnalyze = polygons[0]
                polygon_TP = self.getPointsCount(pointType=PointType.TruePositive, polygon=polygonToAnalyze)
                polygon_TN = self.getPointsCount(pointType=PointType.TrueNegative, polygon=polygonToAnalyze)
                polygon_FP = self.getPointsCount(pointType=PointType.FalsePositive, polygon=polygonToAnalyze)
                polygon_FN = self.getPointsCount(pointType=PointType.FalseNegative, polygon=polygonToAnalyze)
                
                allStatsDicts= allStatsDicts + self.performStatistics(layer.name,
                                                                      polygon_TP,
                                                                      polygon_TN,
                                                                      polygon_FP,
                                                                      polygon_FN)
                
                
        
        tpCount = self.getPointsCount(pointType=PointType.TruePositive, polygon=None)
        tnCount = self.getPointsCount(pointType=PointType.TrueNegative, polygon=None)
        fpCount = self.getPointsCount(pointType=PointType.FalsePositive, polygon=None)
        fnCount = self.getPointsCount(pointType=PointType.FalseNegative, polygon=None)
        
        allStatsDicts = allStatsDicts + self.performStatistics("Overall Stats",
                                                               tpCount,
                                                                tnCount,
                                                                fpCount,
                                                                fnCount)
        
        self.exportStats(allStatsDicts)

    def performStatistics(self, outputName, tpCount, fpCount, tnCount, fnCount):
        totalPoints = tpCount + fpCount + tnCount + fnCount

        precision = tpCount / (tpCount + fpCount)
        recall = tpCount / (tpCount + fnCount)
        fScore = 2 * ((precision * recall) / (precision + recall))
        accuracy = 100 * ((tpCount + tnCount) / (totalPoints))

        actuallyPostitive = tpCount + fnCount
        actuallyNegative = fpCount + tnCount
        predictedPositive = tpCount + fpCount
        predictedNegative = tnCount + fnCount

        fpr = fpCount / actuallyNegative
        tpr = tpCount / actuallyPostitive
        tnr = tnCount / actuallyNegative
        fnr = fnCount / actuallyPostitive
        pLR = tpr / fpr
        nLR = fnr / tnr
        ppv = tpCount / predictedPositive
        npv = tnCount / predictedNegative
        markedness = ppv + npv - 1
        jaccardIndex = tpCount / (tpCount + fnCount + fpCount)

        baseStats = {
            "Stats Section": outputName,
            "True Positive": tpCount,
            "True Negative": tnCount,
            "False Positive": fpCount,
            "False Negative": fnCount,
        }
        derivedStats = {
            "Precision": precision,
            "Accuracy": accuracy,
            "Recall": recall,
            "F-Score": fScore,
            "True Positive Rate": tpr,
            "False Positive Rate": fpr,
            "True Negative Rate": tnr,
            "False Negative Rate": fnr,
            "Positive Likelihood Ratio": pLR,
            "Negative Likelihood Ratio": nLR,
            "Positive Predictive Value": ppv,
            "Negative Predictive Value": npv,
            "Markedness": markedness,
            "Jaccard Index": jaccardIndex
            # "Matthews correlation coefficient": ,
        }

        print("=============================")
        for key, value in baseStats.items():
            print(f"{key}: {value}")
        print("-   -   -   -   -   -   -   -")
        for key, value in derivedStats.items():
            print(f"{key}: {value}")
        print("=============================")

        return [baseStats, derivedStats]
    
    def exportStats(self, statsDictArray):
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
                for statsDict in statsDictArray:
                    for key, value in statsDict.items():
                        writer.writerow([key, value])
                    writer.writerow(["", ""])
                

    def addPointsLayer(self, pointType: PointType):
        new_points_layer = self.viewer.add_points(name=pointType.name)
        new_points_layer.mode = 'add'
        new_points_layer.face_color = pointType.color
        new_points_layer.current_face_color = pointType.color
        new_points_layer.symbol = "disc"
        new_points_layer.size = 10
        new_points_layer.current_size = 10

        def printCoordinates(event):
            last_point = event.source.data[
                -1
            ]  # Get the last point in the data array
            print(f"The last point added is at coordinates: {last_point}")
            totalNumberOfPoints = self.getPointsCount(pointType=pointType, polygon=None)
            countString = f"{pointType.abbreviation}: {totalNumberOfPoints}"
            print(countString)
            self.changeLabel(named=pointType.name, newValue=countString)

        new_points_layer.events.data.connect(
            printCoordinates
        )  # Connect the callback function

    def changeLabel(self, named: str, newValue: str):
        label = self.findChild(QLabel, named)
        if label is not None:
            label.setText(newValue)
        else:
            print("Label was None")
            print(label)

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

class OutlineRegions(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.addShapesLayerButton = QPushButton("Select Regions of Interest")
        self.addShapesLayerButton.clicked.connect(lambda: self.addShapesLayer())
        self.createMasksAndNewLayersButton = QPushButton("Create Masks and Layers")
        self.createMasksAndNewLayersButton.clicked.connect(lambda: self.createMasksAndNewLayers())
        self.layout().addWidget(self.addShapesLayerButton)
        self.layout().addWidget(self.createMasksAndNewLayersButton)
        self.layout().addStretch()

    def addShapesLayer(self):
        maskDrawingLayer = self.viewer.add_shapes(name="SectionMaskShapes")
        maskDrawingLayer.mode = Mode.ADD_POLYGON


    def createMasksAndNewLayers(self):
        layer_image = self.getImageLayer(named="raw")
        layer_polygon = self.getShapesLayer(named="SectionMaskShapes")
        layer_mask = self.getLabelsLayer(named="mask")

        if layer_image is not None and layer_polygon is not None:
            # Add shape layer with rectangle
            # polygon1 = np.array([[20, 20], [60, 20], [70, 50], [50, 80], [30, 50]])
            # polygon2 = np.array([[120, 120], [160, 120], [170, 150], [150, 180], [130, 150]])
            # polygons = [polygon1, polygon2]
            # layer_polygon = self.viewer.add_shapes(polygons, shape_type='polygon', edge_width=1, edge_color='coral', face_color='royalblue', name='shapes')
            polygons = layer_polygon.data
            for index, polygon in enumerate(polygons):
                # Create a Path object from the polygon points
                poly_path = mplPath.Path(polygon)
                
                # Calculate the bounds of the polygon
                x_min, y_min = np.min(polygon, axis=0)
                x_max, y_max = np.max(polygon, axis=0)


                npImageData = layer_image.data.compute()
                mask = np.zeros_like(npImageData, dtype=np.bool_)

                print(layer_image.data)
                print(npImageData)
                print(mask)
                print(".........")
                zRange = polygon.ndim
                
                rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0])
                rr = np.clip(rr, int(y_min), int(y_max)).astype(int)
                cc = np.clip(cc, int(x_min), int(x_max)).astype(int)
                for z in range(zRange):
                    if z > 1:
                        mask[z, cc, rr] = True
                    else:
                        mask[cc, rr] = True

                isolatedPolygon = self.viewer.add_shapes([polygon],
                                                         shape_type='polygon',
                                                         name=f"section{index}_polygon",
                                                         face_color='#55ffff')
                isolatedPolygon.opacity = 0.2


                sectionedImageData = mask * layer_image.data
                sectionedImageLayer = self.viewer.add_image(sectionedImageData,
                                                            name=f'section{index}_image',
                                                            colormap='green',
                                                            blending='additive')
                sectionedImageLayer.gamma = 2
                sectionedImageLayer.opacity = 0.5

                sectionedMaskData = mask * layer_mask.data
                sectionedMaskLayer = self.viewer.add_labels(sectionedMaskData, name=f'section{index}_mask')

                layer_polygon.visible = False
                layer_mask.visible = False
                layer_image.visible = False

    def getImageLayer(self, named: str) -> Optional[Image]:
        for layer in self.viewer.layers:
            if layer.name.lower() == named.lower() and isinstance(layer, Image):
                return layer

        return None
    
    def getShapesLayer(self, named: str) -> Optional[Image]:
        for layer in self.viewer.layers:
            if layer.name.lower() == named.lower() and isinstance(layer, Shapes):
                return layer

        return None
    
    def getLabelsLayer(self, named: str) -> Optional[Points]:
        for layer in self.viewer.layers:
            if layer.name.lower() == named and isinstance(layer, Labels):
                return layer

        return None


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
