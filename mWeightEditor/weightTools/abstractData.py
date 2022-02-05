# https://github.com/chadmv/cmt/blob/master/scripts/cmt/deform/skinio.py
from __future__ import print_function, absolute_import
from ..Qt.QtWidgets import QApplication
from maya import OpenMaya
import maya.api.OpenMaya as OpenMaya2
from maya import cmds

#from ctypes import c_float, c_int
from .mayaToNumpy import mayaToNumpy, numpyToMaya
import numpy as np
from .utils import GlobalContext, getSoftSelectionValuesNEW, getThreeIndices
import six
from six.moves import range, map


# GLOBAL FUNCTIONS
class DataAbstract(object):
    verbose = False

    def __init__(self, createDisplayLocator=True, mainWindow=None):
        self.mainWindow = mainWindow
        self.isSkinData = False
        self.pointsDisplayTrans = None
        self.shapePath = None
        self.deformedShape = None
        self.vertNeighbors = None


        self.isNurbsSurface = False
        self.isNurbsCurve = False
        self.isLattice = False
        self.isMesh = False

        if createDisplayLocator:
            sel = cmds.ls(sl=True)
            hil = cmds.ls(hilite=True)
            self.createDisplayLocator()
            cmds.select(sel)
            cmds.hilite(hil)

    # locator Functions
    def createDisplayLocator(self, forceSelection=False):
        """ Create the pointsDisplay node that we can connect to
        to draw vertex selection, and paint color highlighting
        """
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

        # If any panels are in object isloation mode, make sure that this locator
        # is part of the isolated group so we can ... ya know ... see it
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
        """ Delete the display locator, and remove the reference to it from the class """
        self.deleteDisplayLocator()
        self.pointsDisplayTrans = None

    def deleteDisplayLocator(self):
        """ Delete the display locator if it exists """
        if not self.pointsDisplayTrans:
            return
        if cmds.objExists(self.pointsDisplayTrans):
            cmds.delete(self.pointsDisplayTrans)

    def connectDisplayLocator(self):
        if not self.pointsDisplayTrans:
            return

        if not cmds.objExists(self.pointsDisplayTrans):
            return 

        self.updateDisplayVerts([])
        if self.isMesh:
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
        if cmds.objExists(self.pointsDisplayTrans):
            pointsDisplayTransChildren = cmds.listRelatives(
                self.pointsDisplayTrans, path=True, type="pointsDisplay"
            )
            if not pointsDisplayTransChildren:
                return
            pointsDisplayNode = pointsDisplayTransChildren[0]
            if rowsSel != []:
                if self.isMesh:
                    selVertices = self.orderMelList(
                        [self.vertices[ind] for ind in rowsSel]
                    )
                    inList = ["vtx[{0}]".format(el) for el in selVertices]
                elif self.isNurbsSurface:
                    inList = []
                    selectedVertices = [self.vertices[ind] for ind in rowsSel]
                    for indVtx in selectedVertices:
                        indexV = indVtx % self.numCVsInV
                        indexU = indVtx / self.numCVsInV
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
        """ Get the deformers that are deforming the passed in object
        It is technically possible for the deformed shape not to be a child of
        `sel`, but it's extremely unlikely

        Arguments:
            sel (str): The object to get the deformer of
            typeOfDeformer (str): The deformer type to look for

        Returns:
            str: The name of the deformer node. Empty string if no deformer found
            str: The name of the deformed shape. Empty string if no shape found
        """
        with GlobalContext(message="getDeformerFromSel", doPrint=self.verbose):
            if not sel:
                return "", ""

            selShape = cmds.ls(sel, objectsOnly=True)[0]
            if cmds.ls(selShape, transforms=True):  # if it's a transform get the shape
                selShape = cmds.listRelatives(
                    selShape, shapes=True, path=True, noIntermediate=True
                )
                selShape = selShape[0] if selShape else ""
            if not cmds.ls(selShape, shapes=True):
                return "", ""
            hist = cmds.listHistory(
                selShape, levels=0, pruneDagObjects=True, interestLevel=True
            )
            if typeOfDeformer is not None and hist:
                deformers = cmds.ls(hist, type=typeOfDeformer)
                if deformers:
                    theDeformer = deformers[0]
                    theDeformedShape = cmds.ls(
                        cmds.listHistory(theDeformer, allFuture=True, future=True), type="shape"
                    )
                    return theDeformer, theDeformedShape[0]
            return "", selShape

    def getSoftSelectionVertices(self, inputVertices=None):
        """ Get the current soft selection weights, or the weights of a passed
        set of vertices on the current deformed shape, and store that data
        on the class
        """
        if inputVertices is not None:
            res = inputVertices
        else:
            dicOfSel = getSoftSelectionValuesNEW()
            res = dicOfSel.get(self.deformedShape_longName, [])

        if isinstance(res, tuple):
            self.vertices, self.verticesWeight = res
            self.sortedIndices = np.argsort(self.verticesWeight)[::-1]
            self.opposite_sortedIndices = np.argsort(self.sortedIndices)
            # do the sorting
            self.vertices = self.vertices[self.sortedIndices]
            self.verticesWeight = self.verticesWeight[self.sortedIndices]
        else:
            self.vertices = res
            self.verticesWeight = [1.0] * len(self.vertices)
            self.sortedIndices = list(range(len(self.vertices)))
            self.opposite_sortedIndices = list(range(len(self.vertices)))

    @staticmethod
    def orderMelListValues(vertsIndicesWeights):
        """ Convert the index/weight pairs into [range/weights] groups
        The range is a (start, stop) *INCLUSIVE* pair, and the weights
        are just a list of weights. If the range is only one long, then
        the range will be an int, and the weight will be a float.

        Doing `zip(range(start, stop+1), weights)` will recreate the
        input list. Gotta handle the single cases though.

        Arguments:
            vertsIndicesWeights (list): A list of index/weight pairs

        Returns:
            list: The list of ranges and their weights as described above
        """
        # Separate the idxs and weights
        vertsIndicesWeights.sort(key=lambda x: x[0])
        idxs, weights = zip(*vertsIndicesWeights)
        idxs = np.array(idxs)

        # Split the array where adjacent numders differ by more than 1
        cuts = np.where(idxs[1:] - idxs[:-1] != 1)[0]
        splits = np.split(idxs, cuts + 1)
        pointer = 0
        out = []

        # Use the split index chunks to make the range/weight pairs
        for sp in splits:
            if len(sp) == 1:
                out.append((sp[0], weights[pointer]))
            else:
                out.append(((sp[0], sp[-1]), weights[pointer: pointer + len(sp)]))
            pointer += len(sp)
        return out

    @staticmethod
    def orderMelList(listInd, onlyStr=True):
        """ Group listInd into compact chunks
        If onlyStr is True, return the string values that would go into
        the brackets in a mel selection

        Arguments:
            listInd (list): A list of integers
            onlyStr (bool): Whether or not to return the mel string representation

        Returns:
            list: A list of *INCLUSIVE* ranges. If onlyStr == True, then those
                ranges will be what would go into a mel selection bracket
                Otherwise, the ranges will be pairs (or singles) of numbers
        """
        listInd = np.array(sorted(listInd))

        # Split the array where adjacent numders differ by more than 1
        cuts = np.where(listInd[1:] - listInd[:-1] != 1)[0]
        splits = np.split(listInd, cuts + 1)
        ranges = [[sp[0]] if len(sp) == 1 else [sp[0], sp[-1]] for sp in splits]
        if onlyStr:
            ranges = [':'.join(map(str, rng)) for rng in ranges]
        return ranges

    # functions for MObjects
    @staticmethod
    def getMObject(nodeName, returnDagPath=True):
        """ Get the MObject or MDagPath from a full path of a mesh shape node

        Arguments:
            nodeName (str): The full path to a mesh shape
            returnDagPath (bool): Whether to return the dag path or MObject

        Returns:
            (MDagPath or MObject): The requested maya api object
        """
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
        """ Store the info about the current shape node onto self """
        self.isNurbsSurface = False
        self.isLattice = False
        self.isMesh = False
        self.isNurbsCurve = False

        self.shapePath = self.getMObject(self.deformedShape)
        if self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface:
            self.isNurbsSurface = True
            MfnSurface = OpenMaya.MFnNurbsSurface(self.shapePath)
            self.numCVsInV = MfnSurface.numCVsInV()
            self.numCVsInU = MfnSurface.numCVsInU()
            self.nbVertices = self.numCVsInV * self.numCVsInU
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
            self.isMesh = True
            self.nbVertices = cmds.polyEvaluate(self.deformedShape, vertex=True)
        elif self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve:
            self.isNurbsCurve = True

    @staticmethod
    def _getLatticePoints(theMObject, cvPoints):
        """ Fill an MPointArray for a lattice defined in theMObject """
        s_util = OpenMaya.MScriptUtil()
        t_util = OpenMaya.MScriptUtil()
        u_util = OpenMaya.MScriptUtil()

        s_util.createFromInt(0)
        t_util.createFromInt(0)
        u_util.createFromInt(0)

        s_ptr = s_util.asIntPtr()
        t_ptr = t_util.asIntPtr()
        u_ptr = u_util.asIntPtr()

        latticeFn = OpenMaya.MFnLattice(theMObject)
        latticeFn.getDivisions(s_ptr, t_ptr, u_ptr)

        s_num = s_util.getInt(s_ptr)
        t_num = t_util.getInt(t_ptr)
        u_num = u_util.getInt(u_ptr)

        cvPoints.setLength(s_num, t_num, u_num)
        idx = 0
        for s in range(s_num):
            for t in range(t_num):
                for u in range(u_num):
                    cvPoints[idx] = latticeFn.point(s, t, u)
                    idx += 1

    def getVerticesShape(self, theMObject):
        """ Get the vertices from the shape """
        cvPoints = OpenMaya.MPointArray()
        if self.isMesh:
            theMesh = OpenMaya.MFnMesh(theMObject)
            theMesh.getPoints(cvPoints)
        elif self.isNurbsCurve:
            crvFn = OpenMaya.MFnNurbsCurve(theMObject)
            crvFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
        elif self.isNurbsSurface:
            surfaceFn = OpenMaya.MFnNurbsSurface(theMObject)
            surfaceFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
        elif self.isLattice:
            self._getLatticePoints(theMObject, cvPoints)

        theVertices = mayaToNumpy(cvPoints)
        theVertices = theVertices[:, :3]
        return np.take(theVertices, self.vertices, axis=0)







    # STARRT HERE



    def getConnectVertices(self):
        """ Get the neighbors of all the vertices in a mesh """
        if self.shapePath.apiType() != OpenMaya.MFn.kMesh:
            return
        if self.verbose:
            print("getConnectVertices")
        theMeshFn = OpenMaya.MFnMesh(self.shapePath)
        vertexCount = OpenMaya.MIntArray()
        vertexList = OpenMaya.MIntArray()
        theMeshFn.getVertices(vertexCount, vertexList)
        vertCount = mayaToNumpy(vertexCount).tolist()
        vertexList = mayaToNumpy(vertexList).tolist()

        self.vertNeighbors = {}
        sumVerts = 0
        for nbVertsInFace in vertCount:
            verticesInPolygon = vertexList[sumVerts: sumVerts + nbVertsInFace]
            for i in range(nbVertsInFace):
                self.vertNeighbors.setdefault(verticesInPolygon[i], []).extend(
                    verticesInPolygon[:i] + verticesInPolygon[i + 1:]
                )
            sumVerts += nbVertsInFace
        theMax = 0
        self.nbNeighBoors = {}
        for vtx, lst in six.iteritems(self.vertNeighbors):
            self.vertNeighbors[vtx] = list(set(lst))
            newMax = len(self.vertNeighbors[vtx])
            self.nbNeighBoors[vtx] = newMax
            if newMax > theMax:
                theMax = newMax
        self.maxNeighboors = theMax
        if self.verbose:
            print("end - getConnectVertices")

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
                indexV = indVtx % self.numCVsInV
                indexU = indVtx / self.numCVsInV
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
                indexV = indVtx % self.numCVsInV
                indexU = indVtx / self.numCVsInV
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
