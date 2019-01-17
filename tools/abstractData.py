# https://github.com/chadmv/cmt/blob/master/scripts/cmt/deform/skinio.py
from Qt.QtWidgets import QApplication
from maya import OpenMayaUI, OpenMaya, OpenMayaAnim
from maya import cmds, mel
from functools import partial

# import shiboken2 as shiboken
import time, datetime

from ctypes import c_float, c_int

import numpy as np
import re
from utils import GlobalContext, getSoftSelectionValuesNEW, getThreeIndices


def isin(element, test_elements, assume_unique=False, invert=False):
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(
        element.shape
    )


###################################################################################
#
#   GLOBAL FUNCTIONS
#
###################################################################################
class DataAbstract(object):
    verbose = False

    def __init__(self, createDisplayLocator=True):
        self.isSkinData = False
        sel = cmds.ls(sl=True)
        hil = cmds.ls(hilite=True)
        if createDisplayLocator:
            self.createDisplayLocator()
        cmds.select(sel)
        cmds.hilite(hil)

    # -----------------------------------------------------------------------------------------------------------
    # locator Functions ----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def createDisplayLocator(self):
        self.pointsDisplayTrans = None
        if not cmds.pluginInfo("blurSkin", query=True, loaded=True):
            cmds.loadPlugin("blurSkin")
        if cmds.ls("MSkinWeightEditorDisplay*"):
            cmds.delete(cmds.ls("MSkinWeightEditorDisplay*"))
        self.pointsDisplayTrans = cmds.createNode("transform", n="MSkinWeightEditorDisplay")

        pointsDisplayNode = cmds.createNode("pointsDisplay", p=self.pointsDisplayTrans)
        """
        nurbsConnected = cmds.createNode ("nurbsSurface", p=self.pointsDisplayTrans)
        curveConnected = cmds.createNode ("nurbsCurve", p=self.pointsDisplayTrans)
        meshConnected = cmds.createNode ("mesh", p=self.pointsDisplayTrans)
        for nd in [nurbsConnected, meshConnected, curveConnected] : 
            cmds.setAttr (nd+".v", False)
            cmds.setAttr (nd+".ihi", False)
        #cmds.connectAttr (nurbsConnected+".worldSpace", pointsDisplayNode+".inGeometry", f=True)
        #cmds.connectAttr (meshConnected+".outMesh", pointsDisplayNode+".inMesh", f=True)
        #cmds.connectAttr (meshConnected+".outMesh", pointsDisplayNode+".inGeometry", f=True)
        """

        cmds.setAttr(pointsDisplayNode + ".pointWidth", 5)
        cmds.setAttr(pointsDisplayNode + ".inputColor", 0.0, 1.0, 1.0)

        """
        for nd in [self.pointsDisplayTrans,pointsDisplayNode, meshConnected, nurbsConnected, curveConnected] : 
            cmds.setAttr (nd+".hiddenInOutliner", True)
        """

    def deleteDisplayLocator(self):
        if cmds.objExists(self.pointsDisplayTrans):
            cmds.delete(self.pointsDisplayTrans)

    def connectDisplayLocator(self):
        isMesh = (
            "shapePath" in self.__dict__
            and self.shapePath != None
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
            else:  # self.isNurbsSurface :
                geoType = "nurbsSurface" if self.isNurbsSurface else "nurbsCurve"
                outPlug = ".worldSpace"
                inPlug = ".create"
            if cmds.nodeType(self.deformedShape) != geoType:
                return  # something weird happening, not expected geo

            (pointsDisplayNode,) = cmds.listRelatives(
                self.pointsDisplayTrans, path=True, type="pointsDisplay"
            )
            pdt_geometry = cmds.listRelatives(self.pointsDisplayTrans, path=True, type=geoType)
            if pdt_geometry:
                pdt_geometry = pdt_geometry[0]
                inGeoConn = cmds.listConnections(
                    pointsDisplayNode + ".inGeometry", s=True, d=False, p=True
                )
                if not inGeoConn or inGeoConn[0] != pdt_geometry + outPlug:
                    cmds.connectAttr(
                        pdt_geometry + outPlug, pointsDisplayNode + ".inGeometry", f=True
                    )
                inConn = cmds.listConnections(
                    pdt_geometry + inPlug, s=True, d=False, p=True, scn=True
                )
                if not inConn or inConn[0] != self.deformedShape + outPlug:
                    cmds.connectAttr(self.deformedShape + outPlug, pdt_geometry + inPlug, f=True)
            else:  # for the lattice direct connections -------------------------------------
                inConn = cmds.listConnections(
                    pointsDisplayNode + ".inGeometry", s=True, d=False, p=True, scn=True
                )
                if not inConn or inConn[0] != self.deformedShape + outPlug:
                    cmds.connectAttr(
                        self.deformedShape + outPlug, pointsDisplayNode + ".inGeometry", f=True
                    )

    def updateDisplayVerts(self, rowsSel):
        if isinstance(rowsSel, np.ndarray):
            rowsSel = rowsSel.tolist()
        if "deformedShape" not in self.__dict__:
            return
        if not cmds.objExists(self.deformedShape):
            return

        isMesh = (
            "shapePath" in self.__dict__
            and self.shapePath != None
            and self.shapePath.apiType() == OpenMaya.MFn.kMesh
        )
        if cmds.objExists(self.pointsDisplayTrans):
            (pointsDisplayNode,) = cmds.listRelatives(
                self.pointsDisplayTrans, path=True, type="pointsDisplay"
            )
            if rowsSel != []:
                if isMesh:
                    selVertices = self.orderMelList([self.vertices[ind] for ind in rowsSel])
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
                    selVertices = self.orderMelList([self.vertices[ind] for ind in rowsSel])
                    inList = ["cv[{0}]".format(el) for el in selVertices]
            else:
                inList = []
            if cmds.objExists(pointsDisplayNode):
                cmds.setAttr(
                    pointsDisplayNode + ".inputComponents",
                    *([len(inList)] + inList),
                    type="componentList"
                )

    # -----------------------------------------------------------------------------------------------------------
    # functions utils ------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def getDeformerFromSel(self, sel, typeOfDeformer="skinCluster"):
        with GlobalContext(message="getDeformerFromSel", doPrint=self.verbose):
            if sel:
                # cmds.ls (hilite=True)
                selShape = cmds.ls(sel, objectsOnly=True)[0]
                if cmds.ls(selShape, tr=True):  # if it's a transform get the shape
                    selShape = (
                        cmds.listRelatives(selShape, shapes=True, path=True, noIntermediate=True)
                        or [""]
                    )[0]
                if not cmds.ls(selShape, shapes=True):
                    return "", ""
                hist = cmds.listHistory(selShape, lv=0, pruneDagObjects=True, interestLevel=True)
                if typeOfDeformer != None and hist:
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
            dicOfSel[self.deformedShape_longName] if self.deformedShape_longName in dicOfSel else []
        )
        if inputVertices != None:
            res = inputVertices
        if isinstance(res, tuple):
            self.vertices, self.verticesWeight = res
            arr = np.argsort(self.verticesWeight)
            self.sortedIndices = arr[::-1]
            self.opposite_sortedIndices = np.argsort(self.sortedIndices)
            # do the sorting
            self.vertices = [self.vertices[ind] for ind in self.sortedIndices]
            self.verticesWeight = [self.verticesWeight[ind] for ind in self.sortedIndices]
        else:
            self.vertices = res
            self.verticesWeight = [1.0] * len(self.vertices)
            self.sortedIndices = range(len(self.vertices))
            self.opposite_sortedIndices = range(len(self.vertices))

    def orderMelListValues(self, vertsIndicesWeights):
        vertsIndicesWeights.sort(key=lambda x: x[0])
        # print vertsIndicesWeights
        it = iter(vertsIndicesWeights)
        currentIndex, currentWeight = it.next()
        toReturn = []
        while True:
            try:
                firstIndex, indexPlusOne = currentIndex, currentIndex
                lstWeights = []
                while currentIndex == indexPlusOne:
                    lstWeights.append(currentWeight)
                    indexPlusOne += 1
                    currentIndex, currentWeight = it.next()
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
        # listInd = [49, 60, 61, 62, 80, 81, 82, 83, 100, 101, 102, 103, 113, 119, 120, 121, 138, 139, 140, 158, 159, 178, 179, 198, 230, 231, 250, 251, 252, 270, 271, 272, 273, 274, 291, 292, 293, 319, 320, 321, 360,361,362]
        listInds = []
        listIndString = []
        # listIndStringAndCount = []

        it = iter(listInd)
        currentValue = it.next()
        while True:
            try:
                firstVal = currentValue
                theVal = firstVal
                while currentValue == theVal:
                    currentValue = it.next()
                    theVal += 1
                theVal -= 1
                if firstVal != theVal:
                    toAppend = [firstVal, theVal]
                else:
                    toAppend = [firstVal]
                if onlyStr:
                    listIndString.append(":".join(map(unicode, toAppend)))
                else:
                    listInds.append(toAppend)
                # listIndStringAndCount .append ((theStr,theVal - firstVal + 1))
            except StopIteration:
                if firstVal != theVal:
                    toAppend = [firstVal, theVal]
                else:
                    toAppend = [firstVal]
                if onlyStr:
                    listIndString.append(":".join(map(unicode, toAppend)))
                else:
                    listInds.append(toAppend)
                # listIndStringAndCount .append ((theStr,theVal - firstVal + 1))
                break
        if onlyStr:
            return listIndString
        else:
            return listInds

    # -----------------------------------------------------------------------------------------------------------
    # functions for MObjects  ----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def getMObject(self, nodeName, returnDagPath=True):
        # We expect here the fullPath of a shape mesh
        selList = OpenMaya.MSelectionList()
        OpenMaya.MGlobal.getSelectionListByName(nodeName, selList)
        depNode = OpenMaya.MObject()
        selList.getDependNode(0, depNode)

        if returnDagPath == False:
            return depNode
        mshPath = OpenMaya.MDagPath()
        selList.getDagPath(0, mshPath, depNode)
        return mshPath

    def getShapeInfo(self):
        self.isNurbsSurface = False
        self.isLattice = False

        self.shapePath = self.getMObject(self.deformedShape)
        if (
            self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface
        ):  # cmds.nodeType(shapeName) == 'nurbsSurface':
            self.isNurbsSurface = True
            MfnSurface = OpenMaya.MFnNurbsSurface(self.shapePath)
            self.numCVsInV_ = MfnSurface.numCVsInV()
            self.numCVsInU_ = MfnSurface.numCVsInU()

            self.nbVertices = self.numCVsInV_ * self.numCVsInU_
        elif self.shapePath.apiType() == OpenMaya.MFn.kLattice:  # lattice
            self.isLattice = True
            div_s = cmds.getAttr(self.deformedShape + ".sDivisions")
            div_t = cmds.getAttr(self.deformedShape + ".tDivisions")
            div_u = cmds.getAttr(self.deformedShape + ".uDivisions")
            self.nbVertices = div_s * div_t * div_u
        elif self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve:  # curve
            self.nbVertices = cmds.getAttr(self.deformedShape + ".degree") + cmds.getAttr(
                self.deformedShape + ".spans"
            )
        elif self.shapePath.apiType() == OpenMaya.MFn.kMesh:  # mesh
            self.nbVertices = cmds.polyEvaluate(self.deformedShape, vertex=True)

    def getVerticesShape(self, theMObject):
        if (
            self.shapePath.apiType() == OpenMaya.MFn.kMesh
        ):  # cmds.nodeType(shapeName) == 'nurbsCurve':
            theMesh = OpenMaya.MFnMesh(theMObject)
            lent = theMesh.numVertices() * 3

            cta = (c_float * lent).from_address(int(theMesh.getRawPoints()))
            arr = np.ctypeslib.as_array(cta)
            theVertices = np.reshape(arr, (-1, 3))
        else:
            cvPoints = OpenMaya.MPointArray()
            if (
                self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve
            ):  # cmds.nodeType(shapeName) == 'nurbsCurve':
                crvFn = OpenMaya.MFnNurbsCurve(theMObject)
                crvFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
            elif (
                self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface
            ):  # cmds.nodeType(shapeName) == 'nurbsSurface':
                surfaceFn = OpenMaya.MFnNurbsSurface(theMObject)
                surfaceFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
            pointList = []
            for i in range(cvPoints.length()):
                pointList.append([cvPoints[i][0], cvPoints[i][1], cvPoints[i][2]])
            theVertices = np.array(pointList)
        verticesPosition = np.take(theVertices, self.vertices, axis=0)
        return verticesPosition
        # now subArray of vertices
        # self.origVertices [152]
        # cmds.xform (origShape+".vtx [152]", q=True,ws=True, t=True )

    def getConnectVertices(self):
        """
        dicConnectedVertices = []
        if self.shapePath.apiType() == OpenMaya.MFn.kMesh :
            iterVerts = OpenMaya.MItMeshVertex(self.shapePath)
            arrConn = OpenMaya.MIntArray()
            i = 0
            while not iterVerts.isDone():
                iterVerts.getConnectedVertices(arrConn )
                dicConnectedVertices [i].append (set(arrConn ))
                iterVerts.next ()
                i+=1

        return
        """
        if self.verbose:
            print "getConnectVertices"
        # shapePath = getMObject(None, "restShape")
        theMeshFn = OpenMaya.MFnMesh(self.shapePath)
        vertexCount = OpenMaya.MIntArray()
        vertexList = OpenMaya.MIntArray()

        theMeshFn.getVertices(vertexCount, vertexList)
        # sum it cumulative -------------------------------------------
        # vertCount = np.cumsum (  self.getMIntArray ( vertexCount)).tolist()
        vertCount = self.getMIntArray(vertexCount).tolist()
        vertexList = self.getMIntArray(vertexList).tolist()

        faceVerts = []
        npn = []
        self.vertNeighboors = {}
        sumVerts = 0
        for nbVertsInFace in vertCount:
            QApplication.processEvents()
            verticesInPolygon = vertexList[sumVerts : sumVerts + nbVertsInFace]
            for i in range(nbVertsInFace):
                # self.vertNeighboors.setdefault (verticesInPolygon[i], set( )) .update (verticesInPolygon[0:i],verticesInPolygon[i+1:])
                self.vertNeighboors.setdefault(verticesInPolygon[i], []).extend(
                    verticesInPolygon[0:i] + verticesInPolygon[i + 1 :]
                )
            sumVerts += nbVertsInFace
        theMax = 0
        self.nbNeighBoors = {}
        for vtx, lst in self.vertNeighboors.iteritems():
            QApplication.processEvents()
            self.vertNeighboors[vtx] = list(set(lst))
            newMax = len(self.vertNeighboors[vtx])
            self.nbNeighBoors[vtx] = newMax
            if newMax > theMax:
                theMax = newMax
        # print "theMax ", theMax
        self.maxNeighboors = theMax
        if self.verbose:
            print "end - getConnectVertices"

        """
        shapePath = getMObject("restShape")
        theMeshFn = OpenMaya.MFnMesh (shapePath )

        vertexCount = OpenMaya.MIntArray ()
        vertexList  = OpenMaya.MIntArray ()

        theMeshFn .getVertices(vertexCount, vertexList  )

        vertCount = self.getMIntArray (vertexCount)
        vertexList  = self.getMIntArray (vertexList)

        nbPoly = theMeshFn .numPolygons ()
        nbVertices = theMeshFn .numVertices ()
        # now manipulate the array to get connectedVertices 
        sumVert = 0
        faceVerts = []
        for nbVertices in np.nditer(vertCount):            
            indices = range (sumVert, sumVert+nbVertices)
            faceVerts.append 
            sumVert+=nbVertices [vertexList [indices]]
        """

    def getMIntArray(self, theArr):
        res = OpenMaya.MScriptUtil(theArr)
        util = OpenMaya.MScriptUtil()
        ptr = res.asIntPtr()
        ptCount = theArr.length()
        cta = (c_int * ptCount).from_address(int(ptr))
        out = np.ctypeslib.as_array(cta)
        return np.copy(out)

    # -----------------------------------------------------------------------------------------------------------
    # functions for numpy --------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def printArrayData(self, theArr):  # , theMask) :
        # theArr = self.orig2dArray
        # theMask
        rows = theArr.shape[0]
        cols = theArr.shape[1]
        print "\n"
        for x in range(0, rows):
            toPrint = ""
            sum = 0.0
            for y in range(0, cols):
                # if theMask[x,y] :
                val = theArr[x, y]
                if isinstance(val, np.ma.core.MaskedConstant):
                    toPrint += " --- |"
                else:
                    toPrint += " {0:.1f} |".format(val * 100)
                    # toPrint += str(round(val*100, 1))
                    sum += val
            toPrint += "  -->  {0} ".format(round(sum * 100, 1))
            print toPrint
        print "\n"

    # -----------------------------------------------------------------------------------------------------------
    # get the data --------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def clearData(self):
        self.deformedShape, self.shapeShortName, self.deformedShape_longName = "", "", ""
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
        # for soft order ----------------
        self.sortedIndices = []
        self.opposite_sortedIndices = []

    preSel = ""

    def getDataFromSelection(self, typeOfDeformer="skinCluster", force=True, inputVertices=None):
        with GlobalContext(message="getDataFromSelection", doPrint=self.verbose):
            if inputVertices != None:
                inputVertices = map(int, inputVertices)
            # print inputVertices
            sel = cmds.ls(sl=True)
            theDeformer, deformedShape = self.getDeformerFromSel(sel, typeOfDeformer=typeOfDeformer)
            if not deformedShape or not cmds.objExists(deformedShape):
                return False
            # print "deformedShape -> ",deformedShape
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
            # self.theSkinCluster == theSkinCluster and self.deformedShape == deformedShape
            if not force and isPreloaded:
                return False

            self.shapeShortName = (
                cmds.listRelatives(deformedShape, p=True)[0].split(":")[-1].split("|")[-1]
            )
            splt = self.shapeShortName.split("_")
            if len(splt) > 5:
                self.shapeShortName = "_".join(splt[-7:-4])
            (self.deformedShape_longName,) = cmds.ls(deformedShape, l=True)

            self.deformedShape = deformedShape
            self.theDeformer = theDeformer

            self.raw2dArray = None
            return True

    # -----------------------------------------------------------------------------------------------------------
    # values setting -------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def pruneOnArray(self, theArray, theMask, pruneValue):
        unLock = np.ma.array(theArray.copy(), mask=theMask, fill_value=0)
        np.copyto(theArray, np.full(unLock.shape, 0), where=unLock < pruneValue)

    def pruneWeights(self, pruneValue):
        with GlobalContext(message="pruneWeights", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)

            self.pruneOnArray(new2dArray, self.lockedMask, pruneValue)

            self.setValueInDeformer(new2dArray)
            if self.sub2DArrayToSet != None:
                np.put(self.sub2DArrayToSet, xrange(self.sub2DArrayToSet.size), new2dArray)

    def absoluteVal(self, val):
        with GlobalContext(message="absoluteVal", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)
            absValues = np.full(self.orig2dArray.shape, val)

            np.copyto(new2dArray, absValues, where=self.sumMasks)
            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
            self.setValueInDeformer(new2dArray)

            if self.sub2DArrayToSet != None:
                np.put(self.sub2DArrayToSet, xrange(self.sub2DArrayToSet.size), new2dArray)

    def doAdd(self, val, percent=False, autoPrune=False, average=False, autoPruneValue=0.0001):
        with GlobalContext(message="absoluteVal", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)
            selectArr = np.copy(self.orig2dArray)

            # remaining array -----------------------------------------------------------------------------------------------
            remainingArr = np.copy(self.orig2dArray)
            remainingData = np.ma.array(remainingArr, mask=~self.rmMasks, fill_value=0)
            sum_remainingData = remainingData.sum(axis=1)

            # ---------- first make new mask where remaining values are zero (so no operation can be done ....) -------------
            zeroRemainingIndices = np.flatnonzero(sum_remainingData == 0)
            sumMasksUpdate = self.sumMasks.copy()
            sumMasksUpdate[zeroRemainingIndices, :] = False

            # add the values ------------------------------------------------------------------------------------------------
            theMask = sumMasksUpdate if val < 0.0 else self.sumMasks

            if percent:
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)
                sum_addValues = addValues.sum(axis=1)
                toMult = (sum_addValues + val) / sum_addValues
                addValues = addValues * toMult[:, np.newaxis]
            else:
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0) + val
            # clip it ---------------------------
            addValues = addValues.clip(min=0.0, max=1.0)

            if autoPrune:  # prune values
                self.pruneOnArray(addValues, addValues.mask, addValues.sum(axis=1), autoPruneValue)
            np.copyto(new2dArray, addValues, where=~addValues.mask)
            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
            # self.printArrayData (new2dArray)
            self.setValueInDeformer(new2dArray)

            if self.sub2DArrayToSet != None:
                np.put(self.sub2DArrayToSet, xrange(self.sub2DArrayToSet.size), new2dArray)

    # set Value ------------------------------------------------
    def setValueInDeformer(self):
        pass

    def preSettingValuesFn(self, chunks, actualyVisibleColumns):
        # MASK selection array -----------------------------------
        lstTopBottom = []
        for top, bottom, left, right in chunks:
            lstTopBottom.append(top)
            lstTopBottom.append(bottom)
        self.Mtop, self.Mbottom = min(lstTopBottom), max(lstTopBottom)
        # nb rows selected -------------
        nbRows = self.Mbottom - self.Mtop + 1

        # GET the sub ARRAY ---------------------------------------------------------------------------------
        # self.sub2DArrayToSet = self.raw2dArray [self.Mtop:self.Mbottom+1,]
        self.sub2DArrayToSet = self.display2dArray[
            self.Mtop : self.Mbottom + 1,
        ]
        self.orig2dArray = np.copy(self.sub2DArrayToSet)

        # GET the mask ARRAY ---------------------------------------------------------------------------------
        maskSelection = np.full(self.orig2dArray.shape, False, dtype=bool)
        for top, bottom, left, right in chunks:
            maskSelection[top - self.Mtop : bottom - self.Mtop + 1, left : right + 1] = True

        maskOppSelection = ~maskSelection
        # remove from mask hiddenColumns indices ------------------------------------------------
        hiddenColumns = np.setdiff1d(self.hideColumnIndices, actualyVisibleColumns)
        maskSelection[:, hiddenColumns] = False
        maskOppSelection[:, hiddenColumns] = False

        self.maskColumns = np.full(self.orig2dArray.shape, True, dtype=bool)
        self.maskColumns[:, hiddenColumns] = False

        # get the mask of the locks ------------------------------------------
        self.lockedMask = np.tile(self.lockedColumns, (nbRows, 1))
        lockedRows = [
            ind for ind in range(nbRows) if self.vertices[ind + self.Mtop] in self.lockedVertices
        ]
        self.lockedMask[lockedRows] = True

        # update mask with Locks  ------------------------------------------------------------------------
        self.sumMasks = ~np.add(~maskSelection, self.lockedMask)
        self.nbIndicesSettable = np.sum(self.sumMasks, axis=1)
        self.rmMasks = ~np.add(~maskOppSelection, self.lockedMask)

        # get selected vertices ------------------------------------------------------------------------------
        self.indicesVertices = np.array(
            [self.vertices[indRow] for indRow in xrange(self.Mtop, self.Mbottom + 1)]
        )
        self.indicesWeights = np.array(
            [self.verticesWeight[indRow] for indRow in xrange(self.Mtop, self.Mbottom + 1)]
        )
        if self.softOn and (self.isNurbsSurface or self.isLattice):  # revert indices
            self.indicesVertices = self.indicesVertices[self.opposite_sortedIndices]

    def postSkinSet(self):
        pass

    def getValue(self, row, column):
        return self.display2dArray[row][column]

    # -----------------------------------------------------------------------------------------------------------
    # function to get display  texts ----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def createRowText(self):
        if self.isNurbsSurface:
            self.rowText = []
            for indVtx in self.vertices:
                indexV = indVtx % self.numCVsInV_
                indexU = indVtx / self.numCVsInV_
                # vertInd = self.numCVsInV_ * indexU + indexV
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
            self.rowText = [
                " {0} ".format(ind) for ind in self.vertices
            ]  # map (str, self.vertices)

    # -----------------------------------------------------------------------------------------------------------
    # ------ selection  ----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def getZeroRows(self, selectedColumns):
        res = self.display2dArray[:, selectedColumns]
        myAny = np.any(res, axis=1)
        noneZeroRows = np.where(myAny)[0]
        zeroRows = np.where(~myAny)[0]
        return noneZeroRows

    def selectVertsOfColumns(self, selectedColumns, doSelect=True):
        selectedIndices = self.getZeroRows(selectedColumns)

        # print doSelect,  selectedColumns, selectedIndices
        if doSelect:
            self.selectVerts(selectedIndices)
        else:
            self.updateDisplayVerts(selectedIndices)

    def selectVerts(self, selectedIndices):
        selectedVertices = set([self.vertices[ind] for ind in selectedIndices])
        if not selectedVertices:
            cmds.select(clear=True)
            return
        # print selectedVertices

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
                toSel = ["{0}.vtx[{1}]".format(self.deformedShape, vtx) for vtx in toSel]
            else:  # nurbsCurve
                toSel = ["{0}.cv[{1}]".format(self.deformedShape, vtx) for vtx in toSel]
        # print toSel
        # mel.eval ("select -r " + " ".join(toSel))
        cmds.select(toSel, r=True)

    # -----------------------------------------------------------------------------------------------------------
    # locks ----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def addLockVerticesAttribute(self):
        if not cmds.attributeQuery("lockedVertices", node=self.deformedShape, exists=True):
            cmds.addAttr(self.deformedShape, longName="lockedVertices", dataType="Int32Array")
            # cmds.makePaintable( "mesh", "lockedVertices")

    def getLocksInfo(self):
        self.lockedColumns = []
        self.lockedVertices = []
        # now vertices ------------------
        self.addLockVerticesAttribute()
        self.lockedVertices = cmds.getAttr(self.deformedShape + ".lockedVertices") or []

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
        cmds.setAttr(self.deformedShape + ".lockedVertices", self.lockedVertices, type="Int32Array")

    def isRowLocked(self, row):
        return self.vertices[row] in self.lockedVertices

    def isColumnLocked(self, columnIndex):
        return False

    def isLocked(self, row, columnIndex):
        return self.isColumnLocked(columnIndex) or self.isRowLocked(row)

    # -----------------------------------------------------------------------------------------------------------
    # callBacks ------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def renameCB(self, oldName, newName):
        return
        print "weightEditor call back is Invoked : -{}-  to -{}- ".format(oldName, newName)

    def callUndo(self):
        cmds.Undo()
        pass
