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
    QFrame,
    QVBoxLayout,
    QWidget,
    QCheckBox,
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
    addingFP = 3
    addingFN = 4


class PointType(Enum):
    TruePositive = 0
    FalsePositive = 2
    FalseNegative = 3
    CounterItem = 4

    @property
    def abbreviation(self):
        if self == PointType.TruePositive:
            return "TP"
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
        divider = makeDivider()
        self.layout().addWidget(divider)
        self.addShapesLayerButton = QPushButton("Select Regions of Interest")
        self.addShapesLayerButton.clicked.connect(lambda: addShapesLayer(self))
        self.createMasksAndNewLayersButton = QPushButton("Create Masks and Layers")
        self.createMasksAndNewLayersButton.clicked.connect(lambda: createMasksAndNewLayers(self))
        self.layout().addWidget(self.addShapesLayerButton)
        self.layout().addWidget(self.createMasksAndNewLayersButton)
        self.layout().addStretch()
        self.addPointsLayer(pointType=PointType.CounterItem)

    def addPointsLayer(self, pointType: PointType):
        """Add a new 'Points' type layer and register 'printCoordinates' callback."""
        new_points_layer = self.viewer.add_points(name=pointType.name)
        new_points_layer.mode = 'add'
        new_points_layer.face_color = pointType.color
        new_points_layer.current_face_color = pointType.color
        new_points_layer.symbol = "disc"
        new_points_layer.size = 10
        new_points_layer.current_size = 10

        def printCoordinates(event):
            last_point = event.source.data[-1]
            totalNumberOfPoints = len(event.source.data)
            countString = f"Count: {totalNumberOfPoints}"
            self.totalCountLabel.setText(countString)

        new_points_layer.events.data.connect(printCoordinates)

    def _reset_layer_options(self, event):
        """Reset layer options for each combo in self.layer_combos."""
        for combo in self.layer_combos:
            combo.clear()
            combo.addItems([layer.name for layer in self.viewer.layers if isinstance(layer, Points)])

class PointBasedDataAnalyticsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
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

        self.topFNLabel = QLabel(f"FN: {0}")
        self.topFNLabel.setObjectName(PointType.FalseNegative.name)
        self.layout().addWidget(self.topFNLabel)

        self.peformAnalysisButton = QPushButton("Perform Statistics")
        self.peformAnalysisButton.clicked.connect(self.peformAnalysis)
        self.layout().addWidget(self.peformAnalysisButton)
        # self.peformAnalysisButton.hide()
        divider = makeDivider()
        self.layout().addWidget(divider)
        self.addShapesLayerButton = QPushButton("Select Regions of Interest")
        self.addShapesLayerButton.clicked.connect(lambda: addShapesLayer(self))
        self.createMasksAndNewLayersButton = QPushButton("Create Masks and Layers")
        self.createMasksAndNewLayersButton.clicked.connect(lambda: createMasksAndNewLayers(self))
        self.layout().addWidget(self.addShapesLayerButton)
        self.layout().addWidget(self.createMasksAndNewLayersButton)

        self.addPointsLayer(pointType=PointType.TruePositive)
        self.addPointsLayer(pointType=PointType.FalsePositive)
        self.addPointsLayer(pointType=PointType.FalseNegative)

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
        
        # These are being populated to get the mean stats for the end
        tpCounts = []
        fpCounts = []
        fnCounts = []
        precisions = []
        recalls = []
        fScores = []
        accuracies = []
        jaccardIndexes = []

        for layer in subsectionLayers:
            polygons = layer.data
            if len(polygons) > 0:
                print(f"{layer.name} had {len(polygons)} polygons")
                polygonToAnalyze = polygons[0]
                polygon_TP = getPointsCount(self=self, pointType=PointType.TruePositive, polygon=polygonToAnalyze)
                polygon_FP = getPointsCount(self=self, pointType=PointType.FalsePositive, polygon=polygonToAnalyze)
                polygon_FN = getPointsCount(self=self, pointType=PointType.FalseNegative, polygon=polygonToAnalyze)
                
                stats = self.performStatistics(layer.name, polygon_TP, polygon_FP, polygon_FN) 

                allStatsDicts= allStatsDicts + stats
                tpCounts.append(stats[0]["True Positive"])
                fpCounts.append(stats[0]["False Positive"])
                fnCounts.append(stats[0]["False Negative"])
                precisions.append(stats[1]["Precision"])
                recalls.append(stats[1]["Recall"])
                fScores.append(stats[1]["F-Score"])
                accuracies.append(stats[1]["Accuracy"])
                jaccardIndexes.append(stats[1]["Jaccard Index"])

        # If there is anything in the mean dictionaries
        if len(tpCount) > 0:
            # calculate means
            meanStatsDict = {
                "Stats Section" : "Mean",
                "True Positive": sum(tpCounts) / len(tpCounts),
                "False Positive": sum(fpCounts) / len(fpCounts),
                "False Negative": sum(fnCounts) / len(fnCounts),
                "Precision": sum(precisions) / len(precisions),
                "Recall": sum(recalls) / len(recalls),
                "F-Score": sum(fScores) / len(fScores),
                "Accuracy": sum(accuracies) / len(accuracies),
                "Jaccard Index": sum(jaccardIndexes) / len(jaccardIndexes)
            }

        allStatsDicts.append(meanStatsDict)
        
        # Calculate the overall Stats
        tpCount = getPointsCount(self=self, pointType=PointType.TruePositive, polygon=None)
        fpCount = getPointsCount(self=self, pointType=PointType.FalsePositive, polygon=None)
        fnCount = getPointsCount(self=self, pointType=PointType.FalseNegative, polygon=None)
        
        overallStats = self.performStatistics("Overall Stats",
                                                               tpCount,
                                                                fpCount,
                                                                fnCount)
        allStatsDicts = allStatsDicts + overallStats
        
        self.exportStats(allStatsDicts)

    def performStatistics(self, outputName, tpCount, fpCount, fnCount):
        totalPoints = tpCount + fpCount + fnCount

        precision = tpCount / (tpCount + fpCount)
        recall = tpCount / (tpCount + fnCount)
        fScore = 2 * ((precision * recall) / (precision + recall))
        accuracy = 100 * ((tpCount + fpCount) / (tpCount + fnCount)) # Note this is an approximation of accuracy as outlined in this paper: https://www.aivia-software.com/post/precision-recall-analysis-of-peripheral-nerve-myelinated-axon-counting-pipeline
        
        jaccardIndex = tpCount / (tpCount + fnCount + fpCount)

        baseStats = {
            "Stats Section": outputName,
            "True Positive": tpCount,
            "False Positive": fpCount,
            "False Negative": fnCount,
        }

        derivedStats = {
            "Precision": precision,
            "Recall": recall,
            "F-Score": fScore,
            "Accuracy": accuracy,
            "Jaccard Index": jaccardIndex
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
            totalNumberOfPoints = getPointsCount(self=self, pointType=pointType, polygon=None)
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

def getLayer(self, named=str):
    for layer in self.viewer.layers:
        if layer.name == named:
            return layer
    return None

def selectLayer(self, named=str):
    existingLayer = getLayer(self=self, named=named)
    if existingLayer is not None:    
        for layer in self.viewer.layers:
            if layer.name != named:
                layer.isSelected = False
            else:
                print("Found Layer")
                self.viewer.layers.selection = [layer]
        return existingLayer
    else:
        return None

def addShapesLayer(self):
    layerName = "SectionMaskShapes"
    existingShapeLayer = selectLayer(self=self, named=layerName)
    if existingShapeLayer is None:
        maskDrawingLayer = self.viewer.add_shapes(name=layerName)
        maskDrawingLayer.mode = Mode.ADD_POLYGON
    else:
        existingShapeLayer.mode = Mode.ADD_POLYGON
        existingShapeLayer.visible = True
        getImageLayer(self=self, named="raw").visible = True
        getLabelsLayer(self=self,named="mask").visible = True


def createMasksAndNewLayers(self):
    layer_image = getImageLayer(self=self, named="raw")
    layer_polygon = getShapesLayer(self=self,named="SectionMaskShapes")
    layer_mask = getLabelsLayer(self=self,named="mask")

    if layer_image is not None and layer_polygon is not None:
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
            sectionedMaskLayer.opacity = 0.8

            layer_polygon.visible = False
            layer_mask.visible = False
            layer_image.visible = False

def getPointsCount(self, pointType: PointType, polygon) -> int:

        layer = getPointsLayer(self=self, named=pointType.name.lower())
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
        
def getPointsLayer(self, named: str) -> Optional[Points]:
        """Return first 'Points' type layer matching given name."""
        for layer in self.viewer.layers:
            if layer.name.lower() == named and isinstance(layer, Points):
                return layer
        return None

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

def makeDivider():

    divider = QFrame()
    divider.setFrameShape(QFrame.HLine)
    divider.setFrameShadow(QFrame.Sunken)
    return divider