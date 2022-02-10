# https://github.com/chadmv/cmt/blob/master/scripts/cmt/deform/skinio.py
from __future__ import print_function, absolute_import
from ..Qt.QtWidgets import QApplication
from maya import OpenMaya
import maya.api.OpenMaya as OpenMaya2
from maya import cmds

from ctypes import c_float, c_int

import numpy as np
from .utils import GlobalContext, getSoftSelectionValuesNEW, getThreeIndices
import six
from six.moves import range, map


def isin(element, test_elements, assume_unique=False, invert=False):
    element = np.asarray(element)
    return np.in1d(
        element, test_elements, assume_unique=assume_unique, invert=invert
    ).reshape(element.shape)


# GLOBAL FUNCTIONS
class DataAbstract(object):
    verbose = False

    def __init__(self, createDisplayLocator=True, mainWindow=None):
        self.mainWindow = mainWindow
        self.isSkinData = False
        self.pointsDisplayTrans = None
        self.shapePath = None
        self.deformedShape = None

        if createDisplayLocator:
            sel = cmds.ls(sl=True)
            hil = cmds.ls(hilite=True)
            self.createDisplayLocator()
            cmds.select(sel)
            cmds.hilite(hil)

    # locator Functions
    def createDisplayLocator(self, forceSelection=False):
        self.pointsDisplayTrans = None
        if not cmds.pluginInfo("blurSkin", query=True, loaded=True):
            cmds.loadPlugin("blurSkin")

        if cmds.ls("MSkinWeightEditorDisplay*"):
            cmds.delete(cmds.ls("MSkinWeightEditorDisplay*"))
        self.pointsDisplayTrans = cmds.createNode(
            "transform", name="MSkinWeightEditorDisplay", skipSelect=True
        )

        pointsDisplayNode = cmds.createNode(
            "pointsDisplay", parent=self.pointsDisplayTrans, skipSelect=True
        )
        cmds.setAttr(pointsDisplayNode + ".pointWidth", 5)
        cmds.setAttr(pointsDisplayNode + ".inputColor", 0.0, 1.0, 1.0)

        # add to the Isolate of all
        if forceSelection:
            cmds.select(self.pointsDisplayTrans, add=True)
            # that's added because the isolate doesnt work otherwise, it's dumb I know
        listModelPanels = [
            el
            for el in cmds.getPanel(visiblePanels=True)
            if cmds.getPanel(typeOf=el) == "modelPanel"
        ]
        for thePanel in listModelPanels:
            if cmds.isolateSelect(thePanel, query=True, state=True):
                cmds.isolateSelect(
                    thePanel, addDagObject=self.pointsDisplayTrans
                )  # doesnt work

        if forceSelection:
            cmds.evalDeferred(
                lambda: cmds.select(self.pointsDisplayTrans, deselect=True)
            )
            # that's added because the isolate doesnt work otherwise, it's dumb I know

    def removeDisplayLocator(self):
        self.deleteDisplayLocator()
        self.pointsDisplayTrans = None

    def deleteDisplayLocator(self):
        if not self.pointsDisplayTrans:
            return
        if cmds.objExists(self.pointsDisplayTrans):
            cmds.delete(self.pointsDisplayTrans)

    def connectDisplayLocator(self):
        if not self.pointsDisplayTrans:
            return
        isMesh = (
            self.shapePath is not None
            and self.shapePath.apiType() == OpenMaya.MFn.kMesh
        )
        if cmds.objExists(self.pointsDisplayTrans):
            self.updateDisplayVerts([])
            if isMesh:
                geoType = "mesh"
                outPlug = ".outMesh"
                inPlug = ".inMesh"
            elif self.isLattice:
                geoType = "lattice"
                outPlug = ".worldLattice"
                inPlug = ".latticeInput"
            else:  # self.isNurbsSurface:
                geoType = "nurbsSurface" if self.isNurbsSurface else "nurbsCurve"
                outPlug = ".worldSpace"
                inPlug = ".create"
            if cmds.nodeType(self.deformedShape) != geoType:
                return  # something weird happening, not expected geo

            (pointsDisplayNode,) = cmds.listRelatives(
                self.pointsDisplayTrans, path=True, type="pointsDisplay"
            )
            pdt_geometry = cmds.listRelatives(
                self.pointsDisplayTrans, path=True, type=geoType
            )
            if pdt_geometry:
                pdt_geometry = pdt_geometry[0]
                inGeoConn = cmds.listConnections(
                    pointsDisplayNode + ".inGeometry", s=True, d=False, p=True
                )
                if not inGeoConn or inGeoConn[0] != pdt_geometry + outPlug:
                    cmds.connectAttr(
                        pdt_geometry + outPlug,
                        pointsDisplayNode + ".inGeometry",
                        f=True,
                    )
                inConn = cmds.listConnections(
                    pdt_geometry + inPlug, s=True, d=False, p=True, scn=True
                )
                if not inConn or inConn[0] != self.deformedShape + outPlug:
                    cmds.connectAttr(
                        self.deformedShape + outPlug, pdt_geometry + inPlug, f=True
                    )
            else:  # for the lattice direct connections
                inConn = cmds.listConnections(
                    pointsDisplayNode + ".inGeometry", s=True, d=False, p=True, scn=True
                )
                if not inConn or inConn[0] != self.deformedShape + outPlug:
                    cmds.connectAttr(
                        self.deformedShape + outPlug,
                        pointsDisplayNode + ".inGeometry",
                        f=True,
                    )

    def updateDisplayVerts(self, rowsSel):
        if not self.pointsDisplayTrans:
            return
        if isinstance(rowsSel, np.ndarray):
            rowsSel = rowsSel.tolist()
        if self.deformedShape is None:
            return
        if not cmds.objExists(self.deformedShape):
            return
        isMesh = (
            self.shapePath is not None
            and self.shapePath.apiType() == OpenMaya.MFn.kMesh
        )
        if cmds.objExists(self.pointsDisplayTrans):
            pointsDisplayTransChildren = cmds.listRelatives(
                self.pointsDisplayTrans, path=True, type="pointsDisplay"
            )
            if not pointsDisplayTransChildren:
                return
            pointsDisplayNode = pointsDisplayTransChildren[0]
            if rowsSel != []:
                if isMesh:
                    selVertices = self.orderMelList(
                        [self.vertices[ind] for ind in rowsSel]
                    )
                    inList = ["vtx[{0}]".format(el) for el in selVertices]
                elif self.isNurbsSurface:
                    inList = []
                    selectedVertices = [self.vertices[ind] for ind in rowsSel]
                    for indVtx in selectedVertices:
                        indexV = indVtx % self.numCVsInV_
                        indexU = indVtx / self.numCVsInV_
                        inList.append("cv[{0}][{1}]".format(indexU, indexV))
                elif self.isLattice:
                    inList = []
                    selectedVertices = [self.vertices[ind] for ind in rowsSel]
                    div_s = cmds.getAttr(self.deformedShape + ".sDivisions")
                    div_t = cmds.getAttr(self.deformedShape + ".tDivisions")
                    div_u = cmds.getAttr(self.deformedShape + ".uDivisions")
                    for indVtx in selectedVertices:
                        s, t, u = getThreeIndices(div_s, div_t, div_u, indVtx)
                        inList.append("pt[{0}][{1}][{2}]".format(s, t, u))
                else:
                    selVertices = self.orderMelList(
                        [self.vertices[ind] for ind in rowsSel]
                    )
                    inList = ["cv[{0}]".format(el) for el in selVertices]
            else:
                inList = []
            if cmds.objExists(pointsDisplayNode):
                cmds.setAttr(
                    pointsDisplayNode + ".inputComponents",
                    *([len(inList)] + inList),
                    type="componentList"
                )

    # functions utils
    def getDeformerFromSel(self, sel, typeOfDeformer="skinCluster"):
        with GlobalContext(message="getDeformerFromSel", doPrint=self.verbose):
            if sel:
                selShape = cmds.ls(sel, objectsOnly=True)[0]
                if cmds.ls(selShape, tr=True):  # if it's a transform get the shape
                    selShape = (
                        cmds.listRelatives(
                            selShape, shapes=True, path=True, noIntermediate=True
                        )
                        or [""]
                    )[0]
                if not cmds.ls(selShape, shapes=True):
                    return "", ""
                hist = cmds.listHistory(
                    selShape, lv=0, pruneDagObjects=True, interestLevel=True
                )
                if typeOfDeformer is not None and hist:
                    deformers = cmds.ls(hist, type=typeOfDeformer)
                    if deformers:
                        theDeformer = deformers[0]
                        theDeformedShape = cmds.ls(
                            cmds.listHistory(theDeformer, af=True, f=True), type="shape"
                        )
                        return theDeformer, theDeformedShape[0]
                return "", selShape
            return "", ""

    def getSoftSelectionVertices(self, inputVertices=None):
        dicOfSel = getSoftSelectionValuesNEW()
        res = (
            dicOfSel[self.deformedShape_longName]
            if self.deformedShape_longName in dicOfSel
            else []
        )
        if inputVertices is not None:
            res = inputVertices
        if isinstance(res, tuple):
            self.vertices, self.verticesWeight = res
            arr = np.argsort(self.verticesWeight)
            self.sortedIndices = arr[::-1]
            self.opposite_sortedIndices = np.argsort(self.sortedIndices)
            # do the sorting
            self.vertices = [self.vertices[ind] for ind in self.sortedIndices]
            self.verticesWeight = [
                self.verticesWeight[ind] for ind in self.sortedIndices
            ]
        else:
            self.vertices = res
            self.verticesWeight = [1.0] * len(self.vertices)
            self.sortedIndices = list(range(len(self.vertices)))
            self.opposite_sortedIndices = list(range(len(self.vertices)))

    def orderMelListValues(self, vertsIndicesWeights):
        vertsIndicesWeights.sort(key=lambda x: x[0])
        it = iter(vertsIndicesWeights)
        currentIndex, currentWeight = next(it)
        toReturn = []
        while True:
            try:
                firstIndex, indexPlusOne = currentIndex, currentIndex
                lstWeights = []
                while currentIndex == indexPlusOne:
                    lstWeights.append(currentWeight)
                    indexPlusOne += 1
                    currentIndex, currentWeight = next(it)
                if firstIndex != (indexPlusOne - 1):
                    toAppend = [(firstIndex, (indexPlusOne - 1)), lstWeights]
                else:
                    toAppend = [firstIndex, lstWeights[0]]
                toReturn.append(toAppend)
            except StopIteration:
                if firstIndex != (indexPlusOne - 1):
                    toAppend = [(firstIndex, (indexPlusOne - 1)), lstWeights]
                else:
                    toAppend = [firstIndex, lstWeights[0]]
                toReturn.append(toAppend)
                break
        return toReturn

    def orderMelList(self, listInd, onlyStr=True):
        listInds = []
        listIndString = []

        it = iter(listInd)
        currentValue = next(it)
        while True:
            try:
                firstVal = currentValue
                theVal = firstVal
                while currentValue == theVal:
                    currentValue = next(it)
                    theVal += 1
                theVal -= 1
                if firstVal != theVal:
                    toAppend = [firstVal, theVal]
                else:
                    toAppend = [firstVal]
                if onlyStr:
                    listIndString.append(":".join(map(six.text_type, toAppend)))
                else:
                    listInds.append(toAppend)
            except StopIteration:
                if firstVal != theVal:
                    toAppend = [firstVal, theVal]
                else:
                    toAppend = [firstVal]
                if onlyStr:
                    listIndString.append(":".join(map(six.text_type, toAppend)))
                else:
                    listInds.append(toAppend)
                break
        if onlyStr:
            return listIndString
        else:
            return listInds

    # functions for MObjects
    def getMObject(self, nodeName, returnDagPath=True):
        # We expect here the fullPath of a shape mesh
        selList = OpenMaya.MSelectionList()
        OpenMaya.MGlobal.getSelectionListByName(nodeName, selList)
        depNode = OpenMaya.MObject()
        selList.getDependNode(0, depNode)

        if not returnDagPath:
            return depNode
        mshPath = OpenMaya.MDagPath()
        selList.getDagPath(0, mshPath, depNode)
        return mshPath

    def getShapeInfo(self):
        self.isNurbsSurface = False
        self.isLattice = False

        self.shapePath = self.getMObject(self.deformedShape)
        if self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface:
            self.isNurbsSurface = True
            MfnSurface = OpenMaya.MFnNurbsSurface(self.shapePath)
            self.numCVsInV_ = MfnSurface.numCVsInV()
            self.numCVsInU_ = MfnSurface.numCVsInU()

            self.nbVertices = self.numCVsInV_ * self.numCVsInU_
        elif self.shapePath.apiType() == OpenMaya.MFn.kLattice:
            self.isLattice = True
            div_s = cmds.getAttr(self.deformedShape + ".sDivisions")
            div_t = cmds.getAttr(self.deformedShape + ".tDivisions")
            div_u = cmds.getAttr(self.deformedShape + ".uDivisions")
            self.nbVertices = div_s * div_t * div_u
        elif self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve:
            self.nbVertices = cmds.getAttr(
                self.deformedShape + ".degree"
            ) + cmds.getAttr(self.deformedShape + ".spans")
        elif self.shapePath.apiType() == OpenMaya.MFn.kMesh:
            self.nbVertices = cmds.polyEvaluate(self.deformedShape, vertex=True)

    def getVerticesShape(self, theMObject):
        if self.shapePath.apiType() == OpenMaya.MFn.kMesh:
            theMesh = OpenMaya.MFnMesh(theMObject)
            lent = theMesh.numVertices() * 3

            cta = (c_float * lent).from_address(int(theMesh.getRawPoints()))
            arr = np.ctypeslib.as_array(cta)
            theVertices = np.reshape(arr, (-1, 3))
        else:
            cvPoints = OpenMaya.MPointArray()
            if self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve:
                crvFn = OpenMaya.MFnNurbsCurve(theMObject)
                crvFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
            elif self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface:
                surfaceFn = OpenMaya.MFnNurbsSurface(theMObject)
                surfaceFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
            pointList = []
            for i in range(cvPoints.length()):
                pointList.append([cvPoints[i][0], cvPoints[i][1], cvPoints[i][2]])
            theVertices = np.array(pointList)
        verticesPosition = np.take(theVertices, self.vertices, axis=0)
        return verticesPosition

    def getConnectVertices(self):
        if self.shapePath.apiType() != OpenMaya.MFn.kMesh:
            return
        if self.verbose:
            print("getConnectVertices")
        theMeshFn = OpenMaya.MFnMesh(self.shapePath)
        vertexCount = OpenMaya.MIntArray()
        vertexList = OpenMaya.MIntArray()

        theMeshFn.getVertices(vertexCount, vertexList)
        # sum it cumulative
        vertCount = self.getMIntArray(vertexCount).tolist()
        vertexList = self.getMIntArray(vertexList).tolist()

        self.vertNeighboors = {}
        sumVerts = 0
        for nbVertsInFace in vertCount:
            QApplication.processEvents()
            verticesInPolygon = vertexList[sumVerts: sumVerts + nbVertsInFace]
            for i in range(nbVertsInFace):
                self.vertNeighboors.setdefault(verticesInPolygon[i], []).extend(
                    verticesInPolygon[:i] + verticesInPolygon[i + 1 :]
                )
            sumVerts += nbVertsInFace
        theMax = 0
        self.nbNeighBoors = {}
        for vtx, lst in six.iteritems(self.vertNeighboors):
            QApplication.processEvents()
            self.vertNeighboors[vtx] = list(set(lst))
            newMax = len(self.vertNeighboors[vtx])
            self.nbNeighBoors[vtx] = newMax
            if newMax > theMax:
                theMax = newMax
        self.maxNeighboors = theMax
        if self.verbose:
            print("end - getConnectVertices")

    def getMIntArray(self, theArr):
        res = OpenMaya.MScriptUtil(theArr)
        ptr = res.asIntPtr()
        ptCount = theArr.length()
        cta = (c_int * ptCount).from_address(int(ptr))
        out = np.ctypeslib.as_array(cta)
        return np.copy(out)

    # functions for numpy
    def printArrayData(self, theArr):
        rows = theArr.shape[0]
        cols = theArr.shape[1]
        print("\n")
        for x in range(0, rows):
            toPrint = ""
            sum = 0.0
            for y in range(0, cols):
                val = theArr[x, y]
                if isinstance(val, np.ma.core.MaskedConstant):
                    toPrint += " --- |"
                else:
                    toPrint += " {0:.1f} |".format(val * 100)
                    sum += val
            toPrint += "  -->  {0} ".format(round(sum * 100, 1))
            print(toPrint)
        print("\n")

    # get the data
    def clearData(self):
        self.deformedShape, self.shapeShortName, self.deformedShape_longName = (
            "",
            "",
            "",
        )
        self.theDeformer = ""
        self.isNurbsSurface = False

        self.softIsReallyOn = cmds.softSelect(q=True, softSelectEnabled=True)
        self.softOn = self.softIsReallyOn
        self.prevSoftSel = cmds.softSelect(q=True, softSelectDistance=True)

        self.vertices = []
        self.verticesWeight = []

        self.nbVertices = 0  # used for mesh, curves, nurbs, lattice

        self.rowCount = 0
        self.columnCount = 0
        self.columnsNames = []
        self.shortColumnsNames = []

        self.rowText = []
        self.lockedColumns = []
        self.lockedVertices = []

        self.usedDeformersIndices = []
        self.hideColumnIndices = []
        self.fullShapeIsUsed = False
        # for soft order
        self.sortedIndices = []
        self.opposite_sortedIndices = []

        # undo stack
        self.storeUndo = True
        self.undoValues = None
        self.redoValues = None

    preSel = ""

    def getDataFromSelection(
        self,
        typeOfDeformer="skinCluster",
        force=True,
        inputVertices=None,
        theDeformer=None,
        deformedShape=None,
    ):
        with GlobalContext(message="getDataFromSelection", doPrint=self.verbose):
            if inputVertices is not None:
                inputVertices = list(map(int, inputVertices))
            sel = cmds.ls(sl=True)
            if theDeformer is None or deformedShape is None:
                theDeformer, deformedShape = self.getDeformerFromSel(
                    sel, typeOfDeformer=typeOfDeformer
                )
            self.deformedShape = deformedShape
            self.theDeformer = theDeformer

            if not deformedShape or not cmds.objExists(deformedShape):
                return False
            # check if reloading is necessary
            softOn = cmds.softSelect(q=True, softSelectEnabled=True)
            prevSoftSel = cmds.softSelect(q=True, softSelectDistance=True)
            isPreloaded = (
                self.preSel == sel
                and prevSoftSel == self.prevSoftSel
                and softOn == self.softIsReallyOn
            )

            self.preSel = sel
            self.prevSoftSel = prevSoftSel
            self.softOn = softOn
            self.softIsReallyOn = softOn
            if not force and isPreloaded:
                return False
            self.shapeShortName = (
                cmds.listRelatives(deformedShape, p=True)[0]
                .split(":")[-1]
                .split("|")[-1]
            )
            splt = self.shapeShortName.split("_")
            if len(splt) > 5:
                self.shapeShortName = "_".join(splt[-7:-4])
            (self.deformedShape_longName,) = cmds.ls(deformedShape, long=True)

            self.raw2dArray = None
            return True

    # values setting
    def pruneOnArray(self, theArray, theMask, pruneValue):
        unLock = np.ma.array(theArray.copy(), mask=theMask, fill_value=0)
        np.copyto(theArray, np.full(unLock.shape, 0), where=unLock < pruneValue)

    def pruneWeights(self, pruneValue):
        with GlobalContext(message="pruneWeights", doPrint=self.verbose):
            print("pruneWeights")
            new2dArray = np.copy(self.orig2dArray)

            self.printArrayData(new2dArray)
            self.pruneOnArray(new2dArray, self.lockedMask, pruneValue)
            self.printArrayData(new2dArray)

            self.commandForDoIt(new2dArray)

    def absoluteVal(self, val):
        with GlobalContext(message="absoluteVal", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)
            absValues = np.full(self.orig2dArray.shape, val)

            np.copyto(new2dArray, absValues, where=self.sumMasks)
            if self.softOn:  # mult soft Value
                iw = self.indicesWeights[:, np.newaxis]
                new2dArray = new2dArray * iw + self.orig2dArray * (1.0 - iw)
            self.commandForDoIt(new2dArray)

    def doAdd(
        self, val, percent=False, autoPrune=False, average=False, autoPruneValue=0.0001
    ):
        with GlobalContext(message="absoluteVal", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)
            selectArr = np.copy(self.orig2dArray)

            # remaining array
            remainingArr = np.copy(self.orig2dArray)
            remainingData = np.ma.array(remainingArr, mask=~self.rmMasks, fill_value=0)
            sum_remainingData = remainingData.sum(axis=1)

            # first make new mask where remaining values are zero(so no operation can be done ....)
            zeroRemainingIndices = np.flatnonzero(sum_remainingData == 0)
            sumMasksUpdate = self.sumMasks.copy()
            sumMasksUpdate[zeroRemainingIndices, :] = False

            # add the values
            theMask = sumMasksUpdate if val < 0.0 else self.sumMasks

            if percent:
                maskAry = np.ma.array(selectArr, mask=~theMask, fill_value=0)
                addValues = maskAry + maskAry * val
            else:
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0) + val
            # clip it
            addValues = addValues.clip(min=0.0, max=1.0)

            if autoPrune:  # prune values
                self.pruneOnArray(addValues, addValues.mask, autoPruneValue)
            np.copyto(new2dArray, addValues, where=~addValues.mask)
            if self.softOn:  # mult soft Value
                iw = self.indicesWeights[:, np.newaxis]
                new2dArray = new2dArray * iw + self.orig2dArray * (1.0 - iw)
            self.commandForDoIt(new2dArray)

    def preSettingValuesFn(self, chunks, actualyVisibleColumns):
        # this tells us that before the first set we need to store values for the undo
        self.storeUndo = True

        # MASK selection array
        lstTopBottom = []
        for top, bottom, left, right in chunks:
            lstTopBottom.append(top)
            lstTopBottom.append(bottom)
        self.Mtop, self.Mbottom = min(lstTopBottom), max(lstTopBottom)
        # nb rows selected
        nbRows = self.Mbottom - self.Mtop + 1

        # GET the sub ARRAY
        self.sub2DArrayToSet = self.display2dArray[self.Mtop: self.Mbottom + 1]
        self.orig2dArray = np.copy(self.sub2DArrayToSet)

        # GET the mask ARRAY
        maskSelection = np.full(self.orig2dArray.shape, False, dtype=bool)
        for top, bottom, left, right in chunks:
            maskSelection[
                top - self.Mtop: bottom - self.Mtop + 1, left: right + 1
            ] = True
        maskOppSelection = ~maskSelection
        # remove from mask hiddenColumns indices
        hiddenColumns = np.setdiff1d(self.hideColumnIndices, actualyVisibleColumns)
        if hiddenColumns.any():
            maskSelection[:, hiddenColumns] = False
            maskOppSelection[:, hiddenColumns] = False
        self.maskColumns = np.full(self.orig2dArray.shape, True, dtype=bool)
        if hiddenColumns.any():
            self.maskColumns[:, hiddenColumns] = False
        # get the mask of the locks
        self.lockedMask = np.tile(self.lockedColumns, (nbRows, 1))
        lockedRows = [
            ind
            for ind in range(nbRows)
            if self.vertices[ind + self.Mtop] in self.lockedVertices
        ]
        self.lockedMask[lockedRows] = True

        # update mask with Locks
        self.sumMasks = ~np.add(~maskSelection, self.lockedMask)
        self.nbIndicesSettable = np.sum(self.sumMasks, axis=1)
        self.rmMasks = ~np.add(~maskOppSelection, self.lockedMask)

        # get selected vertices
        self.indicesVertices = np.array(
            [self.vertices[indRow] for indRow in range(self.Mtop, self.Mbottom + 1)]
        )
        self.indicesWeights = np.array(
            [
                self.verticesWeight[indRow]
                for indRow in range(self.Mtop, self.Mbottom + 1)
            ]
        )
        self.subOpposite_sortedIndices = np.argsort(self.indicesVertices)

        if self.softOn and (self.isNurbsSurface or self.isLattice):  # revert indices
            self.indicesVertices = self.indicesVertices[self.opposite_sortedIndices]

    def postSkinSet(self):
        pass

    def getValue(self, row, column):
        return self.display2dArray[row][column]

    def commandForDoIt(self, arrayForSetting):
        self.setValueInDeformer(arrayForSetting)
        if self.sub2DArrayToSet.any():
            np.put(
                self.sub2DArrayToSet, range(self.sub2DArrayToSet.size), arrayForSetting
            )

    def getChunksFromVertices(self, listVertices):
        verts = self.vertices
        vertsIndices = [
            verts.index(vertId) for vertId in listVertices if vertId in verts
        ]
        if not vertsIndices:
            return None
        selVertices = self.orderMelList(vertsIndices, onlyStr=False)
        chunks = []
        for coupleVtx in selVertices:
            if len(coupleVtx) == 1:
                startRow, endRow = coupleVtx[0], coupleVtx[0]
            else:
                startRow, endRow = coupleVtx
            chunks.append((startRow, endRow, 0, self.columnCount))
        return chunks

    def getFullChunks(self):
        return [(0, self.rowCount - 1, 0, self.columnCount)]

    # function to get display texts
    def createRowText(self):
        if self.isNurbsSurface:
            self.rowText = []
            for indVtx in self.vertices:
                indexV = indVtx % self.numCVsInV_
                indexU = indVtx / self.numCVsInV_
                self.rowText.append(" {0} - {1} ".format(indexU, indexV))
        elif self.isLattice:
            self.rowText = []
            div_s = cmds.getAttr(self.deformedShape + ".sDivisions")
            div_t = cmds.getAttr(self.deformedShape + ".tDivisions")
            div_u = cmds.getAttr(self.deformedShape + ".uDivisions")
            for indVtx in self.vertices:
                s, t, u = getThreeIndices(div_s, div_t, div_u, indVtx)
                self.rowText.append(" {0} - {1} - {2} ".format(s, t, u))
        else:
            self.rowText = [" {0} ".format(ind) for ind in self.vertices]

    # selection
    def getZeroRows(self, selectedColumns):
        res = self.display2dArray[:, selectedColumns]
        myAny = np.any(res, axis=1)
        noneZeroRows = np.where(myAny)[0]
        return noneZeroRows

    def selectVertsOfColumns(self, selectedColumns, doSelect=True):
        selectedIndices = self.getZeroRows(selectedColumns)
        if doSelect:
            self.selectVerts(selectedIndices)
        else:
            self.updateDisplayVerts(selectedIndices)

    def selectVerts(self, selectedIndices):
        selectedVertices = set([self.vertices[ind] for ind in selectedIndices])
        if not selectedVertices:
            cmds.select(clear=True)
            return

        if self.isNurbsSurface:
            toSel = []
            for indVtx in selectedVertices:
                indexV = indVtx % self.numCVsInV_
                indexU = indVtx / self.numCVsInV_
                toSel += ["{0}.cv[{1}][{2}]".format(self.deformedShape, indexU, indexV)]
        elif self.isLattice:
            toSel = []
            div_s = cmds.getAttr(self.deformedShape + ".sDivisions")
            div_t = cmds.getAttr(self.deformedShape + ".tDivisions")
            div_u = cmds.getAttr(self.deformedShape + ".uDivisions")
            prt = (
                cmds.listRelatives(self.deformedShape, p=True, path=True)[0]
                if cmds.nodeType(self.deformedShape) == "lattice"
                else self.deformedShape
            )
            for indVtx in self.vertices:
                s, t, u = getThreeIndices(div_s, div_t, div_u, indVtx)
                toSel += ["{0}.pt[{1}][{2}][{3}]".format(prt, s, t, u)]
        else:
            toSel = self.orderMelList(selectedVertices, onlyStr=True)
            if cmds.nodeType(self.deformedShape) == "mesh":
                toSel = [
                    "{0}.vtx[{1}]".format(self.deformedShape, vtx) for vtx in toSel
                ]
            else:  # nurbsCurve
                toSel = ["{0}.cv[{1}]".format(self.deformedShape, vtx) for vtx in toSel]
        cmds.select(toSel, r=True)

    # locks
    def addLockVerticesAttribute(self):
        if not cmds.attributeQuery(
            "lockedVertices", node=self.deformedShape, exists=True
        ):
            cmds.addAttr(
                self.deformedShape, longName="lockedVertices", dataType="Int32Array"
            )

    def getLocksInfo(self):
        self.lockedColumns = []
        self.lockedVertices = []
        # now vertices
        if self.theDeformer != "":
            self.addLockVerticesAttribute()
        att = self.deformedShape + ".lockedVertices"
        if cmds.objExists(att):
            self.lockedVertices = cmds.getAttr(att) or []
        else:
            self.lockedVertices = []
        self.lockedColumns = [False] * self.columnCount

    def unLockRows(self, selectedIndices):
        self.lockRows(selectedIndices, doLock=False)

    def lockRows(self, selectedIndices, doLock=True):
        lockVtx = cmds.getAttr(self.deformedShape + ".lockedVertices") or []
        lockVtx = set(lockVtx)

        selectedVertices = set([self.vertices[ind] for ind in selectedIndices])
        if doLock:
            lockVtx.update(selectedVertices)
        else:
            lockVtx.difference_update(selectedVertices)

        self.lockedVertices = sorted(list(lockVtx))
        cmds.setAttr(
            self.deformedShape + ".lockedVertices",
            self.lockedVertices,
            type="Int32Array",
        )

    def isRowLocked(self, row):
        return self.vertices[row] in self.lockedVertices

    def isColumnLocked(self, columnIndex):
        return False

    def isLocked(self, row, columnIndex):
        return self.isColumnLocked(columnIndex) or self.isRowLocked(row)

    # callBacks
    def renameCB(self, oldName, newName):
        return

    # do and Undo
    def undoRedoFunction(self, arraySetting, sub2DArrayToSet):
        print("! undoRedoFunction !")
        self.setValueInDeformer(arraySetting)
        if sub2DArrayToSet.any():
            np.put(sub2DArrayToSet, range(sub2DArrayToSet.size), arraySetting)


# UNDO REDO FUNCTIONS
class DataQuickSet(object):
    def __init__(
        self,
        undoArgs,
        redoArgs,
        mainWindow=None,
        isSkin=False,
        inListVertices=None,
        influenceIndices=None,
        shapePath=None,
        sknFn=None,
        theSkinCluster=None,
        userComponents=None,
    ):
        self.undoArgs = undoArgs
        self.redoArgs = redoArgs
        self.mainWindow = mainWindow
        self.isSkin = isSkin

        self.inListVertices = inListVertices
        self.influenceIndices = influenceIndices
        self.shapePath = shapePath
        self.sknFn = sknFn
        self.theSkinCluster = theSkinCluster
        self.userComponents = userComponents

        self.blurSkinNode = None
        self.normalizeWeights = None

    def doIt(self):
        pass

    def redoIt(self):
        if not self.isSkin:
            self.setValues(*self.redoArgs)
        else:
            self.blurSkinNode = self.disConnectBlurskinDisplay(self.theSkinCluster)
            self.normalizeWeights = cmds.getAttr(
                self.theSkinCluster + ".normalizeWeights"
            )
            self.setSkinValue(*self.redoArgs)
            self.postSkinSet(self.theSkinCluster, self.inListVertices)
        self.refreshWindow()

    def undoIt(self):
        if not self.isSkin:
            self.setValues(*self.undoArgs)
        else:
            self.blurSkinNode = self.disConnectBlurskinDisplay(self.theSkinCluster)
            self.normalizeWeights = cmds.getAttr(
                self.theSkinCluster + ".normalizeWeights"
            )
            self.setSkinValue(*self.undoArgs)
            self.postSkinSet(self.theSkinCluster, self.inListVertices)
        self.refreshWindow()

    def refreshWindow(self):
        if self.mainWindow:
            try:
                self.mainWindow.refreshBtn()
            except Exception:
                import traceback

                traceback.print_exc()
                print("Exception Occured, and was ignored")
                return

    def setValues(self, attsValues):
        if not attsValues:
            return
        for att, vertsIndicesWeights in attsValues:
            MSel = OpenMaya2.MSelectionList()
            MSel.add(att)

            plg2 = MSel.getPlug(0)
            for indVtx, value in vertsIndicesWeights:
                plg2.elementByLogicalIndex(indVtx).setFloat(value)

    def disConnectBlurskinDisplay(self, theSkinCluster):
        if cmds.objExists(theSkinCluster):
            inConn = cmds.listConnections(
                theSkinCluster + ".input[0].inputGeometry",
                destination=False,
                type="blurSkinDisplay",
            )
            if inConn:
                blurSkinNode = inConn[0]
                inConn = cmds.listConnections(
                    theSkinCluster + ".weightList",
                    destination=False,
                    plugs=True,
                    type="blurSkinDisplay",
                )
                if inConn:
                    cmds.disconnectAttr(inConn[0], theSkinCluster + ".weightList")
                return blurSkinNode
        return ""

    def postSkinSet(self, theSkinCluster, inListVertices):
        cmds.setAttr(theSkinCluster + ".normalizeWeights", self.normalizeWeights)
        if inListVertices and self.blurSkinNode and cmds.objExists(self.blurSkinNode):
            cmds.setAttr(
                self.blurSkinNode + ".inputComponents",
                *([len(inListVertices)] + inListVertices),
                type="componentList"
            )

    def setSkinValue(self, newArray):
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", 0)

        normalize = False
        UndoValues = OpenMaya.MDoubleArray()
        self.sknFn.setWeights(
            self.shapePath,
            self.userComponents,
            self.influenceIndices,
            newArray,
            normalize,
            UndoValues,
        )
