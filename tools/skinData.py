# https://github.com/chadmv/cmt/blob/master/scripts/cmt/deform/skinio.py

from maya import OpenMayaUI, OpenMaya, OpenMayaAnim

# import shiboken2 as shiboken
import time, datetime

from ctypes import c_double
from maya import cmds

import numpy as np
from utils import GlobalContext


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
    def __init__(self):
        self.AllWght = []
        self.usedDeformersIndices = []
        self.theSkinCluster = ""
        self.vertices = []
        self.driverNames = []
        self.shortDriverNames = []
        self.skinningMethod = ""
        self.normalizeWeights = []

        self.lockedColumns = []
        self.lockedVertices = []

        self.rowCount = 0
        self.columnCount = 0

        self.usedDeformersIndices = []
        self.hideColumnIndices = []
        self.meshIsUsed = False

        self.UNDOstack = []

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

    def getIndicesFromSelection(self, sel):
        indices = [el.split(".vtx[")[-1][:-1] for el in sel if ".vtx" in el]
        allIndices = set()
        for index in indices:
            if ":" in index:
                nmbs = map(int, index.split(":"))
                allIndices.update(range(nmbs[0], nmbs[1] + 1))
            else:
                allIndices.add(int(index))
        return sorted(list(allIndices))

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

    def prepareValuesforSetSkinData(self, chunks):
        chunk = chunks[0]
        top, bottom, left, right = chunk

        selectedRows = xrange(top, bottom + 1)
        indicesVertices = [self.vertices[indRow] for indRow in selectedRows]

        # GET the sub ARRAY -------------------------------------------------------
        # self.sub2DArrayToSet = self.raw2dArray [top:bottom+1,left:right+1]
        self.sub2DArrayToSet = self.raw2dArray[
            top : bottom + 1,
        ]
        self.orig2dArray = self.sub2DArrayToSet.copy()

        # MAKING the SUB Array ----------------------------------------------------
        selectedColumns = xrange(left, right + 1)
        condArray = isin(
            range(self.nbDrivers), self.usedDeformersIndices, assume_unique=True, invert=False
        )
        condArray[:left] = False
        condArray[right + 1 :] = False

        # theNamesOfSelectedDeformers = [self.driverNames [ind] for ind, val in enumerate(condArray) if val]
        # print theNamesOfSelectedDeformers

        self.maskArray = np.tile(condArray, (len(selectedRows), 1))

        # NOW Prepare for settingSkin Cluster -------------------------------------
        self.influenceIndices = OpenMaya.MIntArray()
        self.influenceIndices.setLength(self.nbDrivers)
        for i in xrange(self.nbDrivers):
            self.influenceIndices.set(i, i)

        componentType = OpenMaya.MFn.kMeshVertComponent
        fnComponent = OpenMaya.MFnSingleIndexedComponent()
        self.userComponents = fnComponent.create(componentType)
        for ind in indicesVertices:
            fnComponent.addElement(ind)

        lengthArray = self.nbDrivers * (bottom - top + 1)

        self.newArray = OpenMaya.MDoubleArray()
        self.newArray.setLength(lengthArray)

        # set normalize FALSE --------------------------------------------------------
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", 0)

    def setSkinData(self, val):
        with GlobalContext(message="prepareSkinfn numpy"):
            selectedValues = np.copy(self.orig2dArray)
            np.putmask(selectedValues, ~self.maskArray, 0)

            remainingValues = np.copy(self.orig2dArray)
            np.putmask(remainingValues, self.maskArray, 0)

            row_sumsSelected = selectedValues.sum(axis=1)
            row_sumsRemaining = remainingValues.sum(axis=1)

            # --------------------------------------------------------------------
            # with GlobalContext (message = "prepareSkinfn numpy"):
            new2dArray = np.copy(self.orig2dArray)
            # addValues ------------------------
            selectedValues + val
            # np.putmask(new2dArray ,self.maskArray , new2dArray+val)

            # get number selected columns ------------------------------------
            nbColumnsSelected = np.count_nonzero(self.maskArray[0])
            # get total value added       ------------------------------------
            totalAdded = nbColumnsSelected * val
            # remove from non zero other columns -----------------------------
            # self.maskArray.
            # np.putmask(new2dArray ,~self.maskArray , new2dArray-subVal)
            # clamp 0 1 --------------------------------------------------
            new2dArray = new2dArray.clip(min=0, max=1.0)
            # normalize ------------------------------------------------
            row_sums = new2dArray.sum(axis=1)
            new2dArrayDiv = new2dArray / row_sums[:, np.newaxis]
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArrayDiv,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def postSkinSet(self):
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", self.normalizeWeights)
        self.storeUndoStack()

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
        self, new2dArrayDiv, sub2DArrayToSet, userComponents, influenceIndices, shapePath, sknFn
    ):
        """
        # quick settingWeights ------------------------------
        doubles = new2dArrayDiv.flatten()
        count = new2dArrayDiv.size
        util = OpenMaya.MScriptUtil()
        util.createFromList(doubles, count)
        doublePtr = util.asDoublePtr()
        newArray = OpenMaya.MDoubleArray(doublePtr, count)
        """
        # way slower -------------------------------
        # for ind, el in enumerate(new2dArrayDiv.flat) : self.newArray.set( el, ind)

        # WAY WAY slower -------------------------------
        """
        with GlobalContext (message = "prepareValuesforSetSkinData"):        
            new2dArrayDiv = np.copy (self.dataOfSkin.orig2dArray   )
            count = new2dArrayDiv.size    
            #util = OpenMaya.MScriptUtil()
            res = OpenMaya.MScriptUtil( self.dataOfSkin.newArray)
            doublePtr = res.asDoublePtr()
            for i, val in enumerate(new2dArrayDiv.flat): res.setDoubleArray (doublePtr, i, val )
            newArray = OpenMaya.MDoubleArray(doublePtr, count)

        """
        arrayForSetting = np.copy(new2dArrayDiv)
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

        np.put(sub2DArrayToSet, xrange(sub2DArrayToSet.size), new2dArrayDiv)

    def callUndo(self):
        if self.UNDOstack:
            print "UNDO"
            undoArgs = self.UNDOstack.pop()
            self.actuallySetValue(*undoArgs)
        else:
            print "No more undo"

    """
    def setSkinDataOld (self, val) :
        newArray = OpenMaya.MDoubleArray()
        newArray.copy (self.OrigWeights)
        nbColumns = len (self.usedDeformersIndices)

        #for indVtx, vtx in enumerate (self.indicesVertices) : 
        for indVtx, indVtxInRawArray in enumerate (self.selectedRows) : 
            sumVal = 0
            prevSumVal = 0
            for selectedCol in self.userSelectedColumnsIndices : 
                selCol = int(selectedCol)
                indInArray = indVtx*nbColumns+selCol

                origVal = self.OrigWeights [indInArray]
                newVal = np.round (origVal+val, 5)
                if newVal < 0. : newVal = 0.0
                #newVal = np.clip(origVal+val,0,1)
                prevSumVal += origVal
                sumVal += newVal
                newArray.set (newVal, indInArray)
            
            # Normalize -------------------------------------
            if sumVal>1.0 : 
                toMult = 1./sumVal
                for selectedCol in self.userSelectedColumnsIndices : 
                    selCol = int(selectedCol)
                    indInArray = indVtx*nbColumns+selCol

                    newVal = newArray [indInArray]
                    newArray.set (newVal * toMult, indInArray)

                for selectedCol in self.notSelColumnsIndices : 
                    selCol = int(selectedCol)
                    indInArray = indVtx*nbColumns+selCol
                    newArray.set (0, indInArray)
            else : 
                prevToDiv = 1.-prevSumVal
                if prevToDiv :
                    toMult = (1.-sumVal) / prevToDiv
                    globalSum = sumVal
                    for selectedCol in self.notSelColumnsIndices : 
                        selCol = int(selectedCol)
                        indInArray = indVtx*nbColumns+selCol

                        origVal = self.OrigWeights [indInArray]
                        newVal = np.round (origVal*toMult, 5)

                        if ( globalSum + newVal ) >1.0 : newVal = 1.0 - globalSum
                        globalSum += newVal
                        newArray.set (newVal, indInArray)

            # set rawSkinValues ------------------------
            for selectedCol, colBigArr in enumerate (self.usedDeformersIndices) : 
                indInBigArray = indVtxInRawArray*self.nbDrivers+int(colBigArr)
                indInArray = indVtx*nbColumns+int(selectedCol)

                newVal = newArray [indInArray]
                self.rawSkinValues.set (newVal, indInBigArray)
        # store old values
        oldValues  = OpenMaya.MDoubleArray()
        normalize = True
        self.sknFn.setWeights( self.shapePath              ,
                               self.userComponents         ,
                               self.influenceIndices       ,
                               newArray                    ,
                               normalize                   ,
                               oldValues                   ) 

        return
    """

    def exposeSkinData(self, inputSkinCluster, indices=[]):
        self.skinClusterObj = self.getMObject(inputSkinCluster, returnDagPath=False)
        self.sknFn = OpenMayaAnim.MFnSkinCluster(self.skinClusterObj)

        jointPaths = OpenMaya.MDagPathArray()
        self.sknFn.influenceObjects(jointPaths)

        self.shapePath = OpenMaya.MDagPath()
        self.sknFn.getPathAtIndex(0, self.shapePath)
        shapeName = self.shapePath.fullPathName()
        vertexCount = 0

        componentType = OpenMaya.MFn.kMeshVertComponent
        if cmds.nodeType(shapeName) == "nurbsCurve":
            componentType = OpenMaya.MFn.kCurveCVComponent
            crvFn = OpenMaya.MFnNurbsCurve(self.shapePath)
            cvPoints = OpenMaya.MPointArray()

            crvFn.getCVs(cvPoints, OpenMaya.MSpace.kObject)
            vertexCount = cvPoints.length()
        else:
            mshFn = OpenMaya.MFnMesh(self.shapePath)
            vertexCount = mshFn.numVertices()
        fnComponent = OpenMaya.MFnSingleIndexedComponent()
        self.fullComponent = fnComponent.create(componentType)

        if not indices:
            fnComponent.setCompleteData(vertexCount)
        else:
            for ind in indices:
                fnComponent.addElement(ind)
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
                    theDeformedMesh = cmds.ls(
                        cmds.listHistory(skinCluster, af=True, f=True), type="mesh"
                    )

                    return skinCluster, theDeformedMesh[0]
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
        with GlobalContext(message="convertingSkinValues"):
            cta = (c_double * lent).from_address(int(ptr))
            arr = np.ctypeslib.as_array(cta)
            self.raw2dArray = np.copy(arr)
            self.raw2dArray = np.reshape(self.raw2dArray, (-1, self.nbDrivers))
        # now find the zeroColumns ------------------------------------
        myAny = np.any(self.raw2dArray, axis=0)
        self.usedDeformersIndices = np.where(myAny)[0]
        self.hideColumnIndices = np.where(~myAny)[0]

        ################################################################################################################################################################

    def getAllData(self):
        sel = cmds.ls(sl=True)
        theSkinCluster, deformedShape = self.getSkinClusterFromSel(sel)
        if not theSkinCluster:
            return False

        self.theSkinCluster, self.deformedShape = theSkinCluster, deformedShape
        self.raw2dArray = None

        if not theSkinCluster:
            self.usedDeformersIndices = []
            self.hideColumnIndices = []
            self.vertices = []
            self.lockedColumns = []
            self.lockedVertices = []
            self.driverNames = []
            self.shortDriverNames = []
            self.rowCount = 0
            self.columnCount = 0
            self.meshIsUsed = False
            return
        self.driverNames, self.skinningMethod, self.normalizeWeights = self.getSkinClusterValues(
            self.theSkinCluster
        )
        self.shortDriverNames = [el.split(":")[-1].split("|")[-1] for el in self.driverNames]

        self.nbDrivers = len(self.driverNames)

        with GlobalContext(message="rawSkinValues"):
            self.vertices = self.getIndicesFromSelection(sel)
            if not self.vertices:
                self.vertices = cmds.getAttr(
                    "{0}.weightList".format(self.theSkinCluster), multiIndices=True
                )
                self.meshIsUsed = True
                self.rawSkinValues = self.exposeSkinData(self.theSkinCluster)
            else:
                self.rawSkinValues = self.exposeSkinData(self.theSkinCluster, indices=self.vertices)
                self.meshIsUsed = False
            # print "rawSkinValues length : {0}" .format (self.rawSkinValues.length())
        self.hideColumnIndices = []
        self.usedDeformersIndices = range(self.nbDrivers)

        self.rowCount = len(self.vertices)
        self.columnCount = self.nbDrivers

        self.getLocksInfo()
        self.getZeroColumns()
        return True

    def rebuildRawSkin(self):
        if self.meshIsUsed:
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
        return self.lockedColumns[columnIndex]

    def unLockColumns(self, selectedIndices):
        self.lockColumns(selectedIndices, doLock=False)

    def lockColumns(self, selectedIndices, doLock=True):
        for column in selectedIndices:
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

    def selectVerts(self, selectedIndices):
        selectedVertices = set([self.vertices[ind] for ind in selectedIndices])
        toSel = self.orderMelList(selectedVertices, onlyStr=True)
        toSel = ["{0}.vtx[{1}]".format(self.deformedShape, vtx) for vtx in toSel]
        cmds.select(toSel)

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

    def getValue(self, row, column):
        # return self.raw2dArray [row][column] if self.raw2dArray !=None else self.rawSkinValues [row*self.nbDrivers+column]
        return self.raw2dArray[row][column]

    def setValue(self, row, column, value):
        vertexIndex = self.vertices[row]
        deformerName = self.driverNames[column]
        theVtx = "{0}.vtx[{1}]".format(self.deformedShape, vertexIndex)
        print self.theSkinCluster, theVtx, deformerName, value
        cmds.skinPercent(
            self.theSkinCluster, theVtx, transformValue=(deformerName, float(value)), normalize=True
        )
