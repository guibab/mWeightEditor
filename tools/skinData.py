# https://github.com/chadmv/cmt/blob/master/scripts/cmt/deform/skinio.py

from maya import OpenMayaUI, OpenMaya, OpenMayaAnim
from functools import partial

# import shiboken2 as shiboken
import time, datetime

from ctypes import c_double, c_float
from maya import cmds, mel

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
#   SKIN FUNCTIONS
#
###################################################################################
class DataOfSkin(object):
    verbose = False

    def __init__(self, useShortestNames=False, hideZeroColumn=True, createDisplayLocator=True):
        self.useShortestNames = useShortestNames
        self.hideZeroColumn = hideZeroColumn
        self.clearData()
        sel = cmds.ls(sl=True)
        hil = cmds.ls(hilite=True)
        if createDisplayLocator:
            self.createDisplayLocator()
        cmds.select(sel)
        cmds.hilite(hil)

    def createDisplayLocator(self):
        self.pointsDisplayTrans = None
        if not cmds.pluginInfo("blurSkin", query=True, loaded=True):
            cmds.loadPlugin("blurSkin")
        if cmds.ls("MSkinWeightEditorDisplay*"):
            cmds.delete(cmds.ls("MSkinWeightEditorDisplay*"))
        self.pointsDisplayTrans = cmds.createNode("transform", n="MSkinWeightEditorDisplay")
        pointsDisplayNode = cmds.createNode("pointsDisplay", p=self.pointsDisplayTrans)
        nurbsConnected = cmds.createNode("nurbsSurface", p=self.pointsDisplayTrans)
        curveConnected = cmds.createNode("nurbsCurve", p=self.pointsDisplayTrans)
        meshConnected = cmds.createNode("mesh", p=self.pointsDisplayTrans)
        for nd in [nurbsConnected, meshConnected, curveConnected]:
            cmds.setAttr(nd + ".v", False)
            cmds.setAttr(nd + ".ihi", False)
        # cmds.connectAttr (nurbsConnected+".worldSpace", pointsDisplayNode+".inGeometry", f=True)
        # cmds.connectAttr (meshConnected+".outMesh", pointsDisplayNode+".inMesh", f=True)
        # cmds.connectAttr (meshConnected+".outMesh", pointsDisplayNode+".inGeometry", f=True)

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

    ###################################################################################
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

    def fixAroundVertices(self, tolerance=3):
        with GlobalContext(message="fixAroundVertices", doPrint=True):
            geometriesOut = OpenMaya.MObjectArray()
            self.sknFn.getOutputGeometry(geometriesOut)
            outMesh = OpenMaya.MFnMesh(geometriesOut[0])

            geometriesIn = OpenMaya.MObjectArray()
            self.sknFn.getInputGeometry(geometriesIn)
            inMesh = OpenMaya.MFnMesh(geometriesIn[0])

            iterVert = OpenMaya.MItMeshVertex(self.shapePath)

            origVerts = OpenMaya.MFloatPointArray()
            destVerts = OpenMaya.MFloatPointArray()
            inMesh.getPoints(origVerts)
            outMesh.getPoints(destVerts)
            theVert = 0

            problemVerts = set()

            while not iterVert.isDone():
                vertices = OpenMaya.MIntArray()
                iterVert.getConnectedVertices(vertices)

                for i in range(vertices.length()):
                    connVert = vertices[i]
                    origDist = origVerts[theVert].distanceTo(origVerts[connVert])
                    destDist = destVerts[theVert].distanceTo(destVerts[connVert])
                    if (destDist / origDist) > tolerance:
                        problemVerts.add(theVert)
                theVert += 1
                iterVert.next()
        problemVerts = list(problemVerts)
        return problemVerts

    def getVerticesOrigShape(self, outPut=False):
        # from ctypes import c_float
        # inMesh,= cmds.listConnections(self.theSkinCluster+".input[0].inputGeometry", s=True, d=False, p=False, c=False, scn=True)
        # origShape = cmds.ls (cmds.listHistory( inMesh), type = "shape") [0]
        # origMesh = OpenMaya.MFnMesh(self.getMObject(origShape,returnDagPath=True))
        geometries = OpenMaya.MObjectArray()
        if outPut:
            self.sknFn.getOutputGeometry(geometries)
        else:
            self.sknFn.getInputGeometry(geometries)
        theMObject = geometries[0]

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

    def prepareValuesforSetSkinData(self, chunks, actualyVisibleColumns):
        # first check if connected  ---------------------------------------------------
        self.getConnectedBlurskinDisplay(disconnectWeightList=True)

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

        # get normalize values  --------------------------------------------------------------------------
        toNormalizeTo = np.ma.array(self.orig2dArray, mask=~self.lockedMask, fill_value=0.0)
        self.toNormalizeToSum = 1.0 - toNormalizeTo.sum(axis=1).filled(0.0)

        # ---------------------------------------------------------------------------------------------
        # NOW Prepare for settingSkin Cluster ---------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        self.influenceIndices = OpenMaya.MIntArray()
        self.influenceIndices.setLength(self.nbDrivers)
        for i in xrange(self.nbDrivers):
            self.influenceIndices.set(i, i)

        self.indicesVertices = np.array(
            [self.vertices[indRow] for indRow in xrange(self.Mtop, self.Mbottom + 1)]
        )
        self.indicesWeights = np.array(
            [self.verticesWeight[indRow] for indRow in xrange(self.Mtop, self.Mbottom + 1)]
        )
        if self.softOn and (self.isNurbsSurface or self.isLattice):  # revert indices
            self.indicesVertices = self.indicesVertices[self.opposite_sortedIndices]
            # self.indicesWeights  = self.indicesWeights  [self.opposite_sortedIndices]
        if self.isNurbsSurface:
            componentType = OpenMaya.MFn.kSurfaceCVComponent
            fnComponent = OpenMaya.MFnDoubleIndexedComponent()
            self.userComponents = fnComponent.create(componentType)

            for indVtx in self.indicesVertices:
                indexV = int(indVtx % self.numCVsInV_)
                indexU = int(indVtx / self.numCVsInV_)
                fnComponent.addElement(indexU, indexV)
        elif self.isLattice:
            componentType = OpenMaya.MFn.kLatticeComponent
            fnComponent = OpenMaya.MFnTripleIndexedComponent()
            self.userComponents = fnComponent.create(componentType)
            div_s = cmds.getAttr(self.deformedShape + ".sDivisions")
            div_t = cmds.getAttr(self.deformedShape + ".tDivisions")
            div_u = cmds.getAttr(self.deformedShape + ".uDivisions")
            for indVtx in self.indicesVertices:
                s, t, v = getThreeIndices(div_s, div_t, div_u, indVtx)
                fnComponent.addElement(s, t, v)
        else:  # single component
            if self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve:
                componentType = OpenMaya.MFn.kCurveCVComponent
            else:
                componentType = OpenMaya.MFn.kMeshVertComponent
            fnComponent = OpenMaya.MFnSingleIndexedComponent()
            self.userComponents = fnComponent.create(componentType)
            for ind in self.indicesVertices:
                fnComponent.addElement(int(ind))
        lengthArray = self.nbDrivers * (bottom - top + 1)
        self.newArray = OpenMaya.MDoubleArray()
        self.newArray.setLength(lengthArray)

        # set normalize FALSE --------------------------------------------------------
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", 0)

    def printArrayData(self, theArr, theMask):
        # theArr = self.orig2dArray
        # theMask
        rows = theArr.shape[0]
        cols = theArr.shape[1]
        for x in range(0, rows):
            toPrint = ""
            sum = 0.0
            for y in range(0, cols):
                if theMask[x, y]:
                    val = theArr[x, y]
                    if isinstance(val, np.ma.core.MaskedConstant):
                        toPrint += " --- |"
                    else:
                        toPrint += " {0:.1f} |".format(val * 100)
                        # toPrint += str(round(val*100, 1))
                        sum += val
            toPrint += "  -->  {0} ".format(round(sum * 100, 1))
            print toPrint

    def normalize(self):
        with GlobalContext(message="normalize", doPrint=True):
            new2dArray = np.copy(self.orig2dArray)
            unLock = np.ma.array(new2dArray.copy(), mask=self.lockedMask, fill_value=0)
            unLock.clip(0, 1)

            sum_unLock = unLock.sum(axis=1)
            unLockNormalized = (
                unLock / sum_unLock[:, np.newaxis] * self.toNormalizeToSum[:, np.newaxis]
            )

            np.copyto(new2dArray, unLockNormalized, where=~self.lockedMask)
            # normalize -----------------------------------------------------------------------
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def pruneOnArray(self, theArray, theMask, arraySumNormalize, pruneValue):
        unLock = np.ma.array(theArray.copy(), mask=theMask, fill_value=0)
        np.copyto(unLock, np.full(unLock.shape, 0), where=unLock < pruneValue)
        # normalize -----------------------------------------------------------------------
        sum_unLock = unLock.sum(axis=1)
        unLockNormalized = unLock / sum_unLock[:, np.newaxis] * arraySumNormalize[:, np.newaxis]

        np.copyto(theArray, unLockNormalized, where=~theMask)

    def pruneWeights(self, pruneValue):
        with GlobalContext(message="pruneWeights", doPrint=True):
            new2dArray = np.copy(self.orig2dArray)
            self.pruneOnArray(new2dArray, self.lockedMask, self.toNormalizeToSum, pruneValue)
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def getArrayOppInfluences(
        self, leftInfluence="*_L_*", rightInfluence="*_R_*", useRealIndices=False
    ):
        leftSpl = leftInfluence.split(" ")
        rightSpl = rightInfluence.split(" ")
        while "" in leftSpl:
            leftSpl.remove("")
        while "" in rightSpl:
            rightSpl.remove("")
        if len(leftSpl) != len(rightSpl):
            return []

        oppDriverNames = {}
        # useRealIndices is for when the array is sparse (some defomers have been deleted)
        count = max(self.indicesJoints) + 1 if useRealIndices else len(self.driverNames)
        driverNames_oppIndices = [-1] * count

        for ind, influence in enumerate(self.driverNames):
            indInfluence = self.indicesJoints[ind] if useRealIndices else ind

            if driverNames_oppIndices[indInfluence] != -1:
                continue
            oppInfluence = influence
            for i, leftSearch in enumerate(leftSpl):
                rightSearch = rightSpl[i].replace("*", ".*")
                leftSearch = leftSpl[i].replace("*", ".*")
                rightReplace = rightSpl[i].replace("*", "")
                leftReplace = leftSpl[i].replace("*", "")

                if re.search(leftSearch, influence, re.IGNORECASE) != None:
                    oppInfluence = influence.replace(leftReplace, rightReplace)
                    break
                elif re.search(rightSearch, influence, re.IGNORECASE) != None:
                    oppInfluence = influence.replace(rightReplace, leftReplace)
                    break
            if oppInfluence in self.driverNames and oppInfluence != influence:
                oppDriverNames[influence] = oppInfluence
                oppDriverNames[oppInfluence] = influence

                oppInfluenceIndex = self.driverNames.index(oppInfluence)
                if useRealIndices:
                    oppInfluenceIndex = self.indicesJoints[oppInfluenceIndex]

                driverNames_oppIndices[indInfluence] = oppInfluenceIndex
                driverNames_oppIndices[oppInfluenceIndex] = indInfluence
            else:
                oppDriverNames[influence] = influence
                driverNames_oppIndices[indInfluence] = indInfluence
        # print oppDriverNames
        # print driverNames_oppIndices
        return driverNames_oppIndices

    def mirrorArray(self, direction, leftInfluence="*_L_*", rightInfluence="*_R_*"):
        prt = (
            cmds.listRelatives(self.deformedShape, path=-True, parent=True)[0]
            if not cmds.nodeType(self.deformedShape) == "transform"
            else self.deformedShape
        )
        for att in ["symmetricVertices", "rightVertices", "leftVertices", "centerVertices"]:
            if not cmds.attributeQuery(att, node=prt, exists=True):
                return
        symmetricVertices = cmds.getAttr(prt + ".symmetricVertices")
        # rightVertices = cmds.getAttr (prt+".rightVertices")
        # leftVertices = cmds.getAttr (prt+".leftVertices")
        # centerVertices = cmds.getAttr (prt+".centerVertices")

        with GlobalContext(message="mirrorArray", doPrint=self.verbose):
            driverNames_oppIndices = self.getArrayOppInfluences(
                leftInfluence=leftInfluence, rightInfluence=rightInfluence
            )
            if not driverNames_oppIndices:
                return

            # skin setting -----------------------------
            componentType = OpenMaya.MFn.kMeshVertComponent
            fnComponent = OpenMaya.MFnSingleIndexedComponent()
            userComponents = fnComponent.create(componentType)
            symVerts = [int(symmetricVertices[vert]) for vert in self.vertices]
            symVertsSorted = sorted(symVerts)
            indicesSort = [symVerts.index(vert) for vert in symVertsSorted]
            # vertices --------------------------------
            for vert in symVertsSorted:
                fnComponent.addElement(int(vert))
            # joints ----------------
            influenceIndices = OpenMaya.MIntArray()
            influenceIndices.setLength(self.nbDrivers)
            for i in xrange(self.nbDrivers):
                influenceIndices.set(i, i)

            # now the weights -----------------------
            new2dArray = np.copy(self.display2dArray)
            # permutation = np.argsort(driverNames_oppIndices)
            # new2dArray = new2dArray[:,permutation]
            new2dArray = new2dArray[:, np.array(driverNames_oppIndices)]
            new2dArray = new2dArray[np.array(indicesSort), :]
            self.softOn = False
            self.actuallySetValue(
                new2dArray, None, userComponents, influenceIndices, self.shapePath, self.sknFn
            )

            # undoArray = np.copy (self.orig2dArray)
            # self.UNDOstack.append ((undoArray, self.sub2DArrayToSet, self.userComponents, self.influenceIndices, self.shapePath, self.sknFn))
        if self.blurSkinNode and cmds.objExists(self.blurSkinNode):
            # set the vertices
            selVertices = self.orderMelList(symVertsSorted)
            inList = ["vtx[{0}]".format(el) for el in selVertices]
            cmds.setAttr(
                self.blurSkinNode + ".inputComponents",
                *([len(inList)] + inList),
                type="componentList"
            )

    def undoSymetry(self):
        tmpUndoValues = OpenMaya.MDoubleArray()
        if self.undoMirrorValues:
            prevValues, userComponents, influenceIndices = self.undoMirrorValues.pop()
            self.sknFn.setWeights(
                self.shapePath, userComponents, influenceIndices, prevValues, False, tmpUndoValues
            )
        else:
            print "NO MORE SYM UNDO"

    def reassignLocally(self):
        # print "reassignLocally"
        with GlobalContext(message="reassignLocally", doPrint=True):
            # 0 get orig shape ----------------------------------------------------------------
            self.origVerticesPosition = self.getVerticesOrigShape()
            self.origVertsPos = self.origVerticesPosition[
                self.Mtop : self.Mbottom + 1,
            ]

            # 2 get deformers origin position (bindMatrixInverse) ---------------------------------------------
            depNode = OpenMaya.MFnDependencyNode(self.skinClusterObj)
            bindPreMatrixArrayPlug = depNode.findPlug("bindPreMatrix", True)
            mObj = OpenMaya.MObject()
            lstDriverPrePosition = []
            lent = bindPreMatrixArrayPlug.numElements()

            # for ind in xrange ( lent ):
            for ind in self.indicesJoints:
                preMatrixPlug = bindPreMatrixArrayPlug.elementByLogicalIndex(ind)
                matFn = OpenMaya.MFnMatrixData(preMatrixPlug.asMObject())
                mat = matFn.matrix().inverse()
                # position = OpenMaya.MPoint(0,0,0)*mat
                position = [OpenMaya.MScriptUtil.getDoubleArrayItem(mat[3], c) for c in xrange(3)]
                lstDriverPrePosition.append(position)
                # [res.x, res.y, res.z]
            lstDriverPrePosition = np.array(lstDriverPrePosition)

            # 3 make an array of distances -----------------------------------------------------------------------
            # compute the vectors from deformer to the point---------
            a_min_b = self.origVertsPos[:, np.newaxis] - lstDriverPrePosition[np.newaxis, :]
            # compute length of the vectors ------
            distArray = np.linalg.norm(a_min_b, axis=2)
            theMask = self.sumMasks
            distArrayMasked = np.ma.array(distArray, mask=~theMask, fill_value=0)
            # sort the columns --------------------------------------------------------------
            sorted_columns_indices = distArrayMasked.argsort(axis=1)

            # now take the 2 closest columns indices (2 first) and do a dot product of the vectors
            closestIndices = sorted_columns_indices[:, :2]

            # make the vector of 2 closest joints ----------------------------
            closestIndex1 = sorted_columns_indices[:, 0]
            closestIndex2 = sorted_columns_indices[:, 1]
            closestDriversVector = (
                lstDriverPrePosition[closestIndex2] - lstDriverPrePosition[closestIndex1]
            )

            # now get the closest vectors to the point
            # closestVectors_2 = a_min_b [np.arange(closestIndices.shape[0])[:, None], closestIndices, :]
            closestVectors_1 = a_min_b[np.arange(closestIndices.shape[0])[:], closestIndex1]
            """
            ClosestDist = distArray [np.arange(closestIndices.shape[0])[:,None], closestIndices]
            ClosestDistNormalize = ClosestDist / ClosestDist.sum(axis=1)[:, np.newaxis]
            """
            # resDot = np.sum (A * B, axis=1) #-- slower

            # now dot product ----------------------------------------------------
            A = closestVectors_1
            B = closestDriversVector
            resDot = np.einsum("ij, ij->i", A, B)  # A.B
            lengthVectorA = np.linalg.norm(A, axis=1)
            lengthVectorB = np.linalg.norm(B, axis=1)

            # we normalize, then  clip for the negative and substract from 1 to reverse the setting
            normalizedDotProduct = resDot / (lengthVectorB * lengthVectorB)
            normalizedDotProduct = normalizedDotProduct.clip(min=0.0, max=1.0)
            ################################################################################################################
            ################################################################################################################
            # FINISH #######################################################################################################
            ################################################################################################################
            ################################################################################################################
            theMask = self.sumMasks
            # now set the values in the array correct if cross product is positive
            addValues = np.full(self.orig2dArray.shape, 0.0)
            addValues[np.arange(closestIndex2.shape[0])[:], closestIndex2] = normalizedDotProduct
            addValues[np.arange(closestIndex1.shape[0])[:], closestIndex1] = (
                1.0 - normalizedDotProduct
            )

            # addValues [np.arange(closestIndices.shape[0])[:,None], closestIndices] = 1.-ClosestDistNormalize
            addValues = np.ma.array(addValues, mask=~theMask, fill_value=0)

            # 4 normalize it  ----------------------------------------------------------------------------------------
            addValuesNormalized = addValues * self.toNormalizeToSum[:, np.newaxis]

            # 5 copy values  ----------------------------------------------------------------------------------------
            new2dArray = np.copy(self.orig2dArray)
            np.copyto(new2dArray, addValuesNormalized, where=theMask)

            # 6 zero rest array--------------------------------------------------------------------------------------
            ZeroVals = np.full(self.orig2dArray.shape, 0.0)
            np.copyto(new2dArray, ZeroVals, where=self.rmMasks)

            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def absoluteVal(self, val):
        with GlobalContext(message="absoluteVal", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)
            selectArr = np.full(self.orig2dArray.shape, val)

            # remaining array -----------------------------------------------------------------------------------------------
            remainingArr = np.copy(self.orig2dArray)
            remainingData = np.ma.array(remainingArr, mask=~self.rmMasks, fill_value=0)
            sum_remainingData = remainingData.sum(axis=1)

            # ---------- first make new mask where remaining values are zero (so no operation can be done ....) -------------
            zeroRemainingIndices = np.flatnonzero(sum_remainingData == 0)
            sumMasksUpdate = self.sumMasks.copy()
            sumMasksUpdate[zeroRemainingIndices, :] = False

            # add the values ------------------------------------------------------------------------------------------------
            theMask = sumMasksUpdate if val == 0.0 else self.sumMasks
            absValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)

            # normalize the sum to the max value unLocked -------------------------------------------------------------------
            sum_absValues = absValues.sum(axis=1)
            absValuesNormalized = (
                absValues / sum_absValues[:, np.newaxis] * self.toNormalizeToSum[:, np.newaxis]
            )
            np.copyto(
                absValues,
                absValuesNormalized,
                where=sum_absValues[:, np.newaxis] > self.toNormalizeToSum[:, np.newaxis],
            )
            if val != 0.0:  # normalize where rest is zero
                np.copyto(
                    absValues, absValuesNormalized, where=sum_remainingData[:, np.newaxis] == 0.0
                )
            sum_absValues = absValues.sum(axis=1)
            # non selected not locked Rest ---------------------------------------------------------------------------------------------
            restVals = self.toNormalizeToSum - sum_absValues
            toMult = restVals / sum_remainingData
            remainingValues = remainingData * toMult[:, np.newaxis]
            # clip it --------------------------------------------------------------------------------------------------------
            remainingValues = remainingValues.clip(min=0.0, max=1.0)
            # renormalize ---------------------------------------------------------------------------------------------------

            # add with the mask ---------------------------------------------------------------------------------------------
            # np.copyto (new2dArray , absValues.filled(0)+remainingValues.filled(0), where = ~self.lockedMask)
            np.copyto(new2dArray, absValues, where=~absValues.mask)
            np.copyto(new2dArray, remainingValues, where=~remainingValues.mask)

            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def setSkinData(
        self, val, percent=False, autoPrune=False, average=False, autoPruneValue=0.0001
    ):
        # if percent : print "percent"
        with GlobalContext(message="setSkinData", doPrint=self.verbose):
            new2dArray = np.copy(self.orig2dArray)
            selectArr = np.copy(self.orig2dArray)
            remainingArr = np.copy(self.orig2dArray)

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

            if not average and percent:  # percent Add
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)
                sum_addValues = addValues.sum(axis=1)
                toMult = (sum_addValues + val) / sum_addValues
                addValues = addValues * toMult[:, np.newaxis]
            elif not average:  # regular add -------------------
                valuesToAdd = val / self.nbIndicesSettable[:, np.newaxis]
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0) + valuesToAdd
            else:  # ----- average ---------
                if self.verbose:
                    print "average"
                theMask = sumMasksUpdate
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)
                sumCols_addValues = addValues.mean(axis=0)
                theTiling = np.tile(sumCols_addValues, (addValues.shape[0], 1))
                addValues = np.ma.array(theTiling, mask=~theMask, fill_value=0)
            addValues = addValues.clip(min=0, max=1.0)
            # normalize the sum to the max value unLocked -------------------------------------------------------------------
            sum_addValues = addValues.sum(axis=1)
            addValuesNormalized = (
                addValues / sum_addValues[:, np.newaxis] * self.toNormalizeToSum[:, np.newaxis]
            )
            np.copyto(
                addValues,
                addValuesNormalized,
                where=sum_addValues[:, np.newaxis] > self.toNormalizeToSum[:, np.newaxis],
            )
            # normalize where rest is zero
            np.copyto(addValues, addValuesNormalized, where=sum_remainingData[:, np.newaxis] == 0.0)
            sum_addValues = addValues.sum(axis=1)

            # non selected not locked Rest ---------------------------------------------------------------------------------------------
            restVals = self.toNormalizeToSum - sum_addValues
            toMult = restVals / sum_remainingData
            remainingValues = remainingData * toMult[:, np.newaxis]
            # clip it --------------------------------------------------------------------------------------------------------
            remainingValues = remainingValues.clip(min=0.0, max=1.0)
            # renormalize ---------------------------------------------------------------------------------------------------
            if autoPrune:
                self.pruneOnArray(
                    remainingValues,
                    remainingValues.mask,
                    remainingValues.sum(axis=1),
                    autoPruneValue,
                )
                self.pruneOnArray(addValues, addValues.mask, addValues.sum(axis=1), autoPruneValue)
            # add with the mask ---------------------------------------------------------------------------------------------
            # np.copyto (new2dArray , addValues.filled(0)+remainingValues.filled(0), where = ~self.lockedMask)
            np.copyto(new2dArray, addValues, where=~addValues.mask)
            np.copyto(new2dArray, remainingValues, where=~remainingValues.mask)
            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def postSkinSet(self):
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", self.normalizeWeights)
        self.storeUndoStack()

        # if connected  ---------------------------------------------------
        if self.blurSkinNode:
            # set the vertices
            if self.indicesVertices.size > 0:
                selVertices = self.orderMelList(self.indicesVertices)
                inList = ["vtx[{0}]".format(el) for el in selVertices]
                if cmds.objExists(self.blurSkinNode):
                    cmds.setAttr(
                        self.blurSkinNode + ".inputComponents",
                        *([len(inList)] + inList),
                        type="componentList"
                    )
            # cmds.evalDeferred (partial(cmds.connectAttr, self.blurSkinNode+".weightList", self.theSkinCluster+".weightList", f=True))

    def storeUndoStack(self):
        # add to the Undo stack -----------------------------------------
        undoArray = np.copy(self.orig2dArray)
        self.UNDOstack.append(
            (
                undoArray,
                self.sub2DArrayToSet,
                self.userComponents,
                self.influenceIndices,
                self.shapePath,
                self.sknFn,
            )
        )

    def actuallySetValue(
        self, theValues, sub2DArrayToSet, userComponents, influenceIndices, shapePath, sknFn
    ):
        with GlobalContext(message="actuallySetValue", doPrint=self.verbose):
            if self.softOn:
                arrayForSetting = np.copy(theValues[self.opposite_sortedIndices])
            else:
                arrayForSetting = np.copy(theValues)
            doubles = arrayForSetting.flatten()
            count = doubles.size
            tempArrayForSize = OpenMaya.MDoubleArray()
            tempArrayForSize.setLength(count)

            res = OpenMaya.MScriptUtil(tempArrayForSize)
            ptr = res.asDoublePtr()

            # Cast the swig double pointer to a ctypes array
            cta = (c_double * count).from_address(int(ptr))
            out = np.ctypeslib.as_array(cta)
            np.copyto(out, doubles)
            newArray = OpenMaya.MDoubleArray(ptr, count)

            # with GlobalContext (message = "sknFn.setWeights"):
            normalize = False
            UndoValues = OpenMaya.MDoubleArray()
            sknFn.setWeights(
                shapePath, userComponents, influenceIndices, newArray, normalize, UndoValues
            )

            # do the stting in the 2dArray -----
            if sub2DArrayToSet != None:
                np.put(sub2DArrayToSet, xrange(sub2DArrayToSet.size), theValues)
                self.computeSumArray()
            else:
                self.undoMirrorValues.append([UndoValues, userComponents, influenceIndices])

    def callUndo(self):
        if self.UNDOstack:
            if self.verbose:
                print "UNDO"
            undoArgs = self.UNDOstack.pop()
            self.actuallySetValue(*undoArgs)
        else:
            if self.verbose:
                print "No more undo"

    def exposeSkinData(self, inputSkinCluster, indices=[]):
        self.skinClusterObj = self.getMObject(inputSkinCluster, returnDagPath=False)
        self.sknFn = OpenMayaAnim.MFnSkinCluster(self.skinClusterObj)

        jointPaths = OpenMaya.MDagPathArray()
        self.sknFn.influenceObjects(jointPaths)
        self.indicesJoints = []
        for i in range(jointPaths.length()):
            ind = self.sknFn.indexForInfluenceObject(jointPaths[i])
            self.indicesJoints.append(ind)
        self.shapePath = OpenMaya.MDagPath()
        self.sknFn.getPathAtIndex(0, self.shapePath)
        shapeName = self.shapePath.fullPathName()
        vertexCount = 0

        fnComponent = OpenMaya.MFnSingleIndexedComponent()
        self.isNurbsSurface = False
        self.isLattice = False
        componentAlreadyBuild = False
        if self.softOn:
            revertSortedIndices = np.array(indices)[self.opposite_sortedIndices]
        else:
            revertSortedIndices = indices
        if (
            self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve
        ):  # cmds.nodeType(shapeName) == 'nurbsCurve':
            componentType = OpenMaya.MFn.kCurveCVComponent
            crvFn = OpenMaya.MFnNurbsCurve(self.shapePath)
            # cvPoints = OpenMaya.MPointArray()
            # crvFn.getCVs(cvPoints,OpenMaya.MSpace.kObject)
            # vertexCount = cvPoints.length()
            vertexCount = crvFn.numCVs()
        elif (
            self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface
        ):  # cmds.nodeType(shapeName) == 'nurbsSurface':
            self.isNurbsSurface = True
            componentAlreadyBuild = True
            componentType = OpenMaya.MFn.kSurfaceCVComponent
            MfnSurface = OpenMaya.MFnNurbsSurface(self.shapePath)
            # cvPoints = OpenMaya.MPointArray()
            # MfnSurface.getCVs(cvPoints,OpenMaya.MSpace.kObject)
            # vertexCount = cvPoints.length()
            self.numCVsInV_ = MfnSurface.numCVsInV()
            numCVsInU_ = MfnSurface.numCVsInU()
            fnComponent = OpenMaya.MFnDoubleIndexedComponent()
            self.fullComponent = fnComponent.create(componentType)
            if not indices:
                fnComponent.setCompleteData(numCVsInU_, self.numCVsInV_)
            else:
                for indVtx in revertSortedIndices:
                    indexV = indVtx % self.numCVsInV_
                    indexU = indVtx / self.numCVsInV_
                    fnComponent.addElement(indexU, indexV)
        elif self.shapePath.apiType() == OpenMaya.MFn.kLattice:  # lattice
            self.isLattice = True
            componentAlreadyBuild = True
            componentType = OpenMaya.MFn.kLatticeComponent
            fnComponent = OpenMaya.MFnTripleIndexedComponent()
            self.fullComponent = fnComponent.create(componentType)
            div_s = cmds.getAttr(shapeName + ".sDivisions")
            div_t = cmds.getAttr(shapeName + ".tDivisions")
            div_u = cmds.getAttr(shapeName + ".uDivisions")
            if not indices:
                fnComponent.setCompleteData(div_s, div_t, div_u)
            else:
                for indVtx in revertSortedIndices:
                    s, t, v = getThreeIndices(div_s, div_t, div_u, indVtx)
                    fnComponent.addElement(s, t, v)
        elif self.shapePath.apiType() == OpenMaya.MFn.kMesh:  # mesh
            componentType = OpenMaya.MFn.kMeshVertComponent
            mshFn = OpenMaya.MFnMesh(self.shapePath)
            vertexCount = mshFn.numVertices()
        else:
            return None
        if not componentAlreadyBuild:
            self.fullComponent = fnComponent.create(componentType)
            if not indices:
                fnComponent.setCompleteData(vertexCount)
            else:
                for ind in revertSortedIndices:
                    fnComponent.addElement(ind)
        #####################################################
        weights = OpenMaya.MDoubleArray()

        intptrUtil = OpenMaya.MScriptUtil()
        intptrUtil.createFromInt(0)
        intPtr = intptrUtil.asUintPtr()

        self.sknFn.getWeights(self.shapePath, self.fullComponent, weights, intPtr)

        return weights

    def addLockVerticesAttribute(self):
        if not cmds.attributeQuery("lockedVertices", node=self.deformedShape, exists=True):
            cmds.addAttr(self.deformedShape, longName="lockedVertices", dataType="Int32Array")
            # cmds.makePaintable( "mesh", "lockedVertices")

    def getSkinClusterFromSel(self, sel):
        if sel:
            hist = cmds.listHistory(sel, lv=0, pruneDagObjects=True)
            if hist:
                skinClusters = cmds.ls(hist, type="skinCluster")
                if skinClusters:
                    skinCluster = skinClusters[0]
                    theDeformedShape = cmds.ls(
                        cmds.listHistory(skinCluster, af=True, f=True), type="shape"
                    )
                    return skinCluster, theDeformedShape[0]
        return "", ""

    def getSkinClusterValues(self, skinCluster):
        driverNames = cmds.skinCluster(skinCluster, q=True, inf=True)
        skinningMethod = cmds.getAttr(skinCluster + ".skinningMethod")
        normalizeWeights = cmds.getAttr(skinCluster + ".normalizeWeights")
        return (driverNames, skinningMethod, normalizeWeights)

    def getZeroColumns(self):
        """
        arr = np.array([])
        lent = self.rawSkinValues.length()
        arr.resize( lent )
        with GlobalContext (message = "convertingSkinValues"):
            for i, val in enumerate (self.rawSkinValues) : arr[i]=val
        self.raw2dArray = np.reshape(arr, (-1, self.nbDrivers))
        """
        """
        # faster -----------------------------------------------
        arr = np.array([])
        lent = self.rawSkinValues.length() 
        arr.resize( lent )
        res = OpenMaya.MScriptUtil( self.rawSkinValues)
        util = maya.OpenMaya.MScriptUtil()
        ptr = res.asDoublePtr()
        with GlobalContext (message = "convertingSkinValues"):    
            for i in xrange (lent): arr[i]=util.getDoubleArrayItem(ptr, i)
        self.raw2dArray = np.reshape(arr, (-1, self.nbDrivers))
        """
        # deadFast ----------------------------------------------------
        res = OpenMaya.MScriptUtil(self.rawSkinValues)
        util = OpenMaya.MScriptUtil()
        ptr = res.asDoublePtr()

        lent = self.rawSkinValues.length()
        with GlobalContext(message="convertingSkinValues", doPrint=self.verbose):
            cta = (c_double * lent).from_address(int(ptr))
            arr = np.ctypeslib.as_array(cta)
            self.raw2dArray = np.copy(arr)
            self.raw2dArray = np.reshape(self.raw2dArray, (-1, self.nbDrivers))
        # ---- reorder --------------------------------------------
        if self.softOn:  # order with indices
            self.display2dArray = self.raw2dArray[self.sortedIndices]
        else:
            self.display2dArray = self.raw2dArray
        # now find the zeroColumns ------------------------------------

        myAny = np.any(self.raw2dArray, axis=0)
        self.usedDeformersIndices = np.where(myAny)[0]
        self.hideColumnIndices = np.where(~myAny)[0]
        self.computeSumArray()

    def computeSumArray(self):
        if self.raw2dArray != None:
            self.sumArray = self.raw2dArray.sum(axis=1)

    def getShortNames(self):
        self.shortDriverNames = []
        for el in self.driverNames:
            shortName = el.split(":")[-1].split("|")[-1]
            if self.useShortestNames and shortName.startswith("Dfm_"):
                splt = shortName.split("_")
                shortName = " ".join(splt[1:])
            self.shortDriverNames.append(shortName)

    def clearData(self):
        self.AllWght = []
        self.theSkinCluster, self.deformedShape, self.shapeShortName = "", "", ""
        self.isNurbsSurface = False
        self.blurSkinNode = ""

        self.softIsReallyOn = cmds.softSelect(q=True, softSelectEnabled=True)
        self.softOn = self.softIsReallyOn
        self.prevSoftSel = cmds.softSelect(q=True, softSelectDistance=True)

        self.vertices = []
        self.verticesWeight = []
        self.driverNames = []
        self.indicesJoints = []
        self.nbDrivers = 0
        self.shortDriverNames = []
        self.rowText = []
        self.skinningMethod = ""
        self.normalizeWeights = []

        # for soft order ----------------
        self.sortedIndices = []
        self.opposite_sortedIndices = []

        self.lockedColumns = []
        self.lockedVertices = []

        self.rowCount = 0
        self.columnCount = 0

        self.usedDeformersIndices = []
        self.hideColumnIndices = []
        self.fullShapeIsUsed = False

        self.UNDOstack = []
        self.undoMirrorValues = []

    def getConnectedBlurskinDisplay(self, disconnectWeightList=False):
        self.blurSkinNode = ""
        if cmds.objExists(self.theSkinCluster):
            inConn = cmds.listConnections(
                self.theSkinCluster + ".input[0].inputGeometry",
                s=True,
                d=False,
                type="blurSkinDisplay",
            )
            if inConn:
                self.blurSkinNode = inConn[0]
                if disconnectWeightList:
                    inConn = cmds.listConnections(
                        self.theSkinCluster + ".weightList",
                        s=True,
                        d=False,
                        p=True,
                        type="blurSkinDisplay",
                    )
                    if inConn:
                        cmds.disconnectAttr(inConn[0], self.theSkinCluster + ".weightList")
                return self.blurSkinNode
        return ""

    preSel = ""

    def getAllData(self, displayLocator=True, getskinWeights=True, force=True, inputVertices=None):
        if inputVertices != None:
            inputVertices = map(int, inputVertices)
        # print inputVertices
        sel = cmds.ls(sl=True)

        theSkinCluster, deformedShape = self.getSkinClusterFromSel(sel)
        if not theSkinCluster:
            return False
        # check if reloading is necessary
        # isPreloaded = theSkinCluster == self.theSkinCluster and   deformedShape == self.deformedShape # same selection as before
        softOn = cmds.softSelect(q=True, softSelectEnabled=True)
        prevSoftSel = cmds.softSelect(q=True, softSelectDistance=True)
        isPreloaded = (
            self.preSel == sel and prevSoftSel == self.prevSoftSel and softOn == self.softIsReallyOn
        )

        self.preSel = sel
        self.prevSoftSel = prevSoftSel
        self.softOn = softOn
        self.softIsReallyOn = softOn
        # self.theSkinCluster == theSkinCluster and self.deformedShape == deformedShape
        if not force and isPreloaded:
            return False

        self.theSkinCluster, self.deformedShape = theSkinCluster, deformedShape
        self.shapeShortName = (
            cmds.listRelatives(deformedShape, p=True)[0].split(":")[-1].split("|")[-1]
        )
        splt = self.shapeShortName.split("_")
        if len(splt) > 5:
            self.shapeShortName = "_".join(splt[-7:-4])

        self.raw2dArray = None

        if not theSkinCluster:
            self.clearData()
            return False
        # get orig vertices -------------------------------
        self.driverNames, self.skinningMethod, self.normalizeWeights = self.getSkinClusterValues(
            self.theSkinCluster
        )
        self.getShortNames()
        self.nbDrivers = len(self.driverNames)

        with GlobalContext(message="rawSkinValues", doPrint=self.verbose):
            dicOfSel = getSoftSelectionValuesNEW()
            res = dicOfSel[self.deformedShape] if self.deformedShape in dicOfSel else []
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
            if not self.vertices:
                self.vertices = cmds.getAttr(
                    "{0}.weightList".format(self.theSkinCluster), multiIndices=True
                )
                self.verticesWeight = [1.0] * len(self.vertices)
                self.sortedIndices = range(len(self.vertices))
                self.opposite_sortedIndices = range(len(self.vertices))
                self.softOn = 0
                self.fullShapeIsUsed = True
                if getskinWeights:
                    self.rawSkinValues = self.exposeSkinData(self.theSkinCluster)
            else:
                if getskinWeights:
                    self.rawSkinValues = self.exposeSkinData(
                        self.theSkinCluster, indices=self.vertices
                    )
                self.fullShapeIsUsed = False
            # print "rawSkinValues length : {0}" .format (self.rawSkinValues.length())
            if not getskinWeights:
                return True
        isMesh = (
            "shapePath" in self.__dict__
            and self.shapePath != None
            and self.shapePath.apiType() == OpenMaya.MFn.kMesh
        )
        # if displayLocator and (isMesh or self.isNurbsSurface) : self.connectDisplayLocator ()
        if displayLocator:
            self.connectDisplayLocator()

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
        self.hideColumnIndices = []
        self.usedDeformersIndices = range(self.nbDrivers)

        self.rowCount = len(self.vertices)
        self.columnCount = self.nbDrivers

        self.getLocksInfo()
        self.getZeroColumns()
        return True

    def smoothSkin(self, selectedIndices, repeat=1, percentMvt=1):
        # cmds.blurSkinCmd (command = "smooth", repeat = self.smoothBTN.precision, percentMvt = self.percentBTN.precision)
        rowsSel = []
        for item in selectedIndices:
            rowsSel += range(item[0], item[1] + 1)
        selectedVertices = sorted([self.vertices[ind] for ind in rowsSel])

        if self.isNurbsSurface:
            listCVsIndices = []
            for indVtx in selectedVertices:
                indexV = indVtx % self.numCVsInV_
                indexU = indVtx / self.numCVsInV_
                listCVsIndices.append((indexU, indexV))
            cmds.blurSkinCmd(
                command="smooth",
                repeat=repeat,
                percentMvt=percentMvt,
                meshName=self.deformedShape,
                listCVsIndices=listCVsIndices,
            )
        elif self.isLattice:
            cmds.blurSkinCmd(command="smooth", repeat=repeat, percentMvt=percentMvt)
        else:
            cmds.blurSkinCmd(
                command="smooth",
                repeat=repeat,
                percentMvt=percentMvt,
                meshName=self.deformedShape,
                listVerticesIndices=selectedVertices,
            )

    def rebuildRawSkin(self):
        if self.fullShapeIsUsed:
            self.rawSkinValues = self.exposeSkinData(self.theSkinCluster)
        else:
            self.rawSkinValues = self.exposeSkinData(self.theSkinCluster, indices=self.vertices)

    def getLocksInfo(self):
        self.lockedColumns = []
        self.lockedVertices = []

        for driver in self.driverNames:
            isLocked = False
            if cmds.attributeQuery("lockInfluenceWeights", node=driver, exists=True):
                isLocked = cmds.getAttr(driver + ".lockInfluenceWeights")
            self.lockedColumns.append(isLocked)
        # now vertices ------------------
        self.addLockVerticesAttribute()
        self.lockedVertices = cmds.getAttr(self.deformedShape + ".lockedVertices") or []

    def isLocked(self, row, columnIndex):
        return self.isColumnLocked(columnIndex) or self.isRowLocked(row)

    def isRowLocked(self, row):
        return self.vertices[row] in self.lockedVertices

    def isColumnLocked(self, columnIndex):
        return columnIndex >= self.nbDrivers or self.lockedColumns[columnIndex]

    def unLockColumns(self, selectedIndices):
        self.lockColumns(selectedIndices, doLock=False)

    def lockColumns(self, selectedIndices, doLock=True):
        for column in selectedIndices:
            if column < self.nbDrivers:
                driver = self.driverNames[column]
                if cmds.objExists(driver + ".lockInfluenceWeights"):
                    cmds.setAttr(driver + ".lockInfluenceWeights", doLock)
                    self.lockedColumns[column] = doLock

    def selectDeformers(self, selectedIndices):
        toSel = [
            self.driverNames[column]
            for column in selectedIndices
            if cmds.objExists(self.driverNames[column])
        ]
        cmds.select(toSel)
        cmds.selectMode(object=True)

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
        if not self.blurSkinNode or not cmds.objExists(self.blurSkinNode):
            self.getConnectedBlurskinDisplay()
        if self.blurSkinNode and cmds.objExists(self.blurSkinNode):
            cmds.setAttr(self.blurSkinNode + ".getLockWeights", True)
            # update

    def getValue(self, row, column):
        return self.display2dArray[row][column] if column < self.nbDrivers else self.sumArray[row]
        # return self.raw2dArray [row][column] if column < self.nbDrivers else self.sumArray [row]

    def setValue(self, row, column, value):
        vertexIndex = self.vertices[row]
        deformerName = self.driverNames[column]
        theVtx = "{0}.vtx[{1}]".format(self.deformedShape, vertexIndex)
        if self.verbose:
            print self.theSkinCluster, theVtx, deformerName, value
        # cmds.skinPercent( self.theSkinCluster,theVtx, transformValue=(deformerName, float (value)), normalize = True)

    def driverNamesChangedCB(self, oldName, newName):
        if oldName in self.driverNames:
            ind = self.driverNames.index(oldName)
            self.driverNames[ind] = newName
            return True
        return False
