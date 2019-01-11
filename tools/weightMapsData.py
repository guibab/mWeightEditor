from maya import OpenMayaUI, OpenMaya, OpenMayaAnim
import maya.api.OpenMaya as OpenMaya2

from maya import cmds, mel
from functools import partial

# import shiboken2 as shiboken
import time, datetime

from ctypes import c_double, c_float

import numpy as np
import re
from utils import GlobalContext, getSoftSelectionValuesNEW, getThreeIndices

from .abstractData import DataAbstract, isin

"""
cmds.getAttr ("blendShape1.inputTarget[0].baseWeights") [0]
cmds.getAttr ("blendShape1.inputTarget[0].baseWeights", mi = True)

values = [.11,.12,.13]
cmds.setAttr ("blendShape1.inputTarget[0].baseWeights[0:2]",*values, size=len(values))
"""


class DataOfBlendShape(DataAbstract):
    verbose = False

    def __init__(self, useShortestNames=False, hideZeroColumn=True, createDisplayLocator=True):
        self.useShortestNames = useShortestNames
        self.hideZeroColumn = hideZeroColumn
        self.clearData()
        super(DataOfBlendShape, self).__init__(createDisplayLocator=createDisplayLocator)

    # -------------------------------------------------------------------------------------------
    # blendShape functions ---------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def getBlendShapesAttributes(self, BSnode, theNodeShape):
        lsGeomsOrig = cmds.blendShape(BSnode, q=True, geometry=True)
        lsGeomsIndicesOrig = cmds.blendShape(BSnode, q=True, geometryIndices=True)

        listAttrs = []
        listAttrShortName = []
        if theNodeShape in lsGeomsOrig:
            # get the index of the node in the blendShape
            inputTarget = lsGeomsIndicesOrig[lsGeomsOrig.index(theNodeShape)]

            listAttrShortName.append("baseWeights")
            listAttrs.append("{}.inputTarget[{}].baseWeights".format(BSnode, inputTarget))

            # get the alias -------------------------------------------------------
            listAlias = cmds.aliasAttr(BSnode, q=True)
            listAliasIndices = cmds.getAttr(
                BSnode + ".inputTarget[{}].inputTargetGroup".format(inputTarget), mi=True
            )

            listAliasNme = (
                zip(listAlias[0::2], listAlias[1::2])
                if listAlias
                else [
                    ("targetWeights_{}".format(i), "weight[{}]".format(i)) for i in listAliasIndices
                ]
            )
            dicIndex = {}
            for el, wght in listAliasNme:
                dicIndex[int(re.findall(r"\b\d+\b", wght)[0])] = el
            # end alias -------------------------------------------------------------

            for channelIndex in listAliasIndices:
                attrShortName = dicIndex[channelIndex]
                attr = "{}.inputTarget[{}].inputTargetGroup[{}].targetWeights".format(
                    BSnode, inputTarget, channelIndex
                )

                listAttrShortName.append(attrShortName)
                listAttrs.append(attr)
            return listAttrShortName, listAttrs
        else:
            return [], []

    def getBlendShapeValues(self):
        nbAttrs = len(self.listAttrs)
        # initialize array at 1.0
        self.raw2dArray = np.full((self.nbVertices, nbAttrs), 1.0)
        """
        indices = [10L,15L,32L]
        values = [0.1,0.3,0.4]
        arr [indices,1] = values
        """
        with GlobalContext():
            for indAtt, att in enumerate(self.listAttrs):
                indices = cmds.getAttr(att, mi=True)
                if indices:
                    values = cmds.getAttr(att)[0]
                    self.raw2dArray[indices, indAtt] = values
        # ---- reorder --------------------------------------------
        if self.softOn:  # order with indices
            self.display2dArray = self.raw2dArray[self.sortedIndices]
        else:
            self.display2dArray = self.raw2dArray

    def setBlendShapeValue(self, att):
        weights = [1.0] * count

        MSel = OpenMaya2.MSelectionList()
        MSel.add(att)

        plg2 = MSel.getPlug(0)
        ids = plg2.getExistingArrayAttributeIndices()
        count = len(ids)

        with GlobalContext():
            for i in xrange(count):
                plg2.elementByLogicalIndex(i).setFloat(weights[i])
        # elementByLogicalIndex  faster than elementByPhysicalIndex

    # -------------------------------------------------------------------------------------------
    # redefine abstract data functions ---------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def clearData(self):
        super(DataOfBlendShape, self).clearData()
        self.BSnode = ""
        self.listAttrShortName, self.listAttrs = [], []

    preSel = ""

    def getAllData(self, displayLocator=True, force=True, inputVertices=None):
        success = self.getDataFromSelection(
            typeOfDeformer="blendShape", force=force, inputVertices=inputVertices
        )
        if not success:
            return False
        else:
            self.BSnode = self.theDeformer

        print self.BSnode
        self.getShapeInfo()
        # get list belndShapes attributes
        self.shortColumnsNames, self.listAttrs = self.getBlendShapesAttributes(
            self.BSnode, self.deformedShape
        )
        # get blendShapes weights values
        self.getBlendShapeValues()

        if displayLocator:
            self.connectDisplayLocator()
        self.getSoftSelectionVertices(inputVertices=inputVertices)

        if not self.vertices:
            self.vertices = range(self.nbVertices)
            self.verticesWeight = [1.0] * len(self.vertices)
            self.sortedIndices = range(len(self.vertices))
            self.opposite_sortedIndices = range(len(self.vertices))
            self.softOn = 0
            self.fullShapeIsUsed = True
        else:
            self.fullShapeIsUsed = False
        self.createRowText()
        self.rowCount = self.nbVertices
        self.columnCount = len(self.listAttrs)

        self.getLocksInfo()
        return True
