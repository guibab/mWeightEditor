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
        self.nbDrivers = 0
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
        selectedVertices = [el for el in sel if ".vtx[" in el]
        indices = [el.split(".vtx[")[-1][:-1] for el in selectedVertices if ".vtx" in el]

        for toSearch in [".f[", ".e["]:
            selection = [el for el in sel if toSearch in el]
            kwargs = {"tv": True}
            if "f" in toSearch:
                kwargs["ff"] = True
            else:
                kwargs["fe"] = True
            convertedVertices = cmds.polyListComponentConversion(selection, **kwargs)
            indices += [el.split(".vtx[")[-1][:-1] for el in convertedVertices if ".vtx" in el]
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

    def prepareValuesforSetSkinData(self, chunks, actualyVisibleColumns):
        # MASK selection array -----------------------------------
        lstTopBottom = []
        for top, bottom, left, right in chunks:
            lstTopBottom.append(top)
            lstTopBottom.append(bottom)
        Mtop, Mbottom = min(lstTopBottom), max(lstTopBottom)
        # nb rows selected -------------
        nbRows = Mbottom - Mtop + 1

        # GET the sub ARRAY ---------------------------------------------------------------------------------
        self.sub2DArrayToSet = self.raw2dArray[
            Mtop : Mbottom + 1,
        ]
        self.orig2dArray = np.copy(self.sub2DArrayToSet)

        # GET the mask ARRAY ---------------------------------------------------------------------------------
        maskSelection = np.full(self.orig2dArray.shape, False, dtype=bool)
        for top, bottom, left, right in chunks:
            maskSelection[top - Mtop : bottom - Mtop + 1, left : right + 1] = True

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
            ind for ind in range(nbRows) if self.vertices[ind + Mtop] in self.lockedVertices
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

        componentType = OpenMaya.MFn.kMeshVertComponent
        fnComponent = OpenMaya.MFnSingleIndexedComponent()
        self.userComponents = fnComponent.create(componentType)

        indicesVertices = [self.vertices[indRow] for indRow in xrange(Mtop, Mbottom + 1)]
        for ind in indicesVertices:
            fnComponent.addElement(ind)

        lengthArray = self.nbDrivers * (bottom - top + 1)

        self.newArray = OpenMaya.MDoubleArray()
        self.newArray.setLength(lengthArray)

        # set normalize FALSE --------------------------------------------------------
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", 0)

        # return
        """
        # apply the mask ----------------------------------------------------------------
        myMaskedData = np.ma.array(self.orig2dArray , mask = ~maskSelection, fill_value = 0 )
        #myMaskedData.mask = np.ma.nomask
        #myMaskedData.mask = lockedMask
        #sumToNormalizeTo = myMaskedData.sum(axis=1)
        val = .2
        arrayofVals = val / nbIndicesSettable[:, np.newaxis]
        #new2dArraySum =  myMaskedData + val / nbIndicesForDivide[:, np.newaxis]


        # print -------------------------------------------------------
        rows = self.orig2dArray.shape[0]
        cols = self.orig2dArray.shape[1]
        for x in range(0, rows):
            toPrint = ""
            sum = 0.0
            for y in range(0, cols):
                if rmMasks [x,y] : 
                    val = self.orig2dArray[x,y] 
                    toPrint +="{0:.1f} ".format(val *100)
                    sum += val
            toPrint += "  -->  {0:.1f} ".format(sum  *100)
            print toPrint 

        #myMaskedData = np.ma.array(self.orig2dArray , mask = ~rmMasks, fill_value = 0 )        
        #res = sumToNormalizeTo [2]
        #isinstance (res ,np.ma.core.MaskedConstant)
        """

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

    def absoluteVal(self, val):
        with GlobalContext(message="absoluteVal", doPrint=True):
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
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def setSkinData(self, val):
        with GlobalContext(message="setSkinData", doPrint=False):
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

            valuesToAdd = val / self.nbIndicesSettable[:, np.newaxis]
            addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0) + valuesToAdd
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

            # add with the mask ---------------------------------------------------------------------------------------------
            # np.copyto (new2dArray , addValues.filled(0)+remainingValues.filled(0), where = ~self.lockedMask)
            np.copyto(new2dArray, addValues, where=~addValues.mask)
            np.copyto(new2dArray, remainingValues, where=~remainingValues.mask)

            """
            row_sumsPrev = myMaskedData .sum(axis=1)
            row_sums = sumValues.sum(axis=1)

            rmv_sums = removeValues .sum(axis=1)
            i = 7
            print row_sums  [i] - row_sumsPrev [i]            
            print sumToNormalizeTo [i] + row_sumsPrev [i]
            print rmv_sums [i] + row_sums [i]

            """
        # set Value ------------------------------------------------
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def prepareValuesforSetSkinDataOLD(self, chunks, actualyVisibleColumns):
        chunk = chunks[0]
        top, bottom, left, right = chunk

        # get selected vertices -----------------------------------------------------------------------------
        selectedRows = xrange(top, bottom + 1)
        indicesVertices = [
            self.vertices[indRow] for indRow in selectedRows
        ]  # if not self.isRowLocked (indRow) ]

        # GET the sub ARRAY ---------------------------------------------------------------------------------
        self.sub2DArrayToSet = self.raw2dArray[
            top : bottom + 1,
        ]
        self.orig2dArray = self.sub2DArrayToSet.copy()

        # MAKING the MASK Array ------------------------------------------------------------------------------
        # first add the shown columns ---
        visibleColumns = np.union1d(self.usedDeformersIndices, actualyVisibleColumns)

        # selectedColumns = xrange(left,right+1)
        # get columns in the visible set --------------------------------
        condArray = isin(range(self.nbDrivers), visibleColumns, assume_unique=True, invert=False)
        condArray[:left] = False
        condArray[right + 1 :] = False

        # theNamesOfSelectedDeformers = [self.driverNames [ind] for ind, val in enumerate(condArray) if val]
        # print theNamesOfSelectedDeformers

        self.maskArray = np.tile(condArray, (len(selectedRows), 1))
        # set at False the locked vertices ---------------------------------------------------------------

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

    def setSkinDataOLD(self, val):
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
            # selectedValues+val
            np.putmask(new2dArray, self.maskArray, new2dArray + val)

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
        self, theValues, sub2DArrayToSet, userComponents, influenceIndices, shapePath, sknFn
    ):
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
        np.put(sub2DArrayToSet, xrange(sub2DArrayToSet.size), theValues)
        self.computeSumArray()

    def callUndo(self):
        if self.UNDOstack:
            print "UNDO"
            undoArgs = self.UNDOstack.pop()
            self.actuallySetValue(*undoArgs)
        else:
            print "No more undo"

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
        self.computeSumArray()

    def computeSumArray(self):
        self.sumArray = self.raw2dArray.sum(axis=1)

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
        return columnIndex >= self.nbDrivers or self.lockedColumns[columnIndex]

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
        return self.raw2dArray[row][column] if column < self.nbDrivers else self.sumArray[row]

    def setValue(self, row, column, value):
        vertexIndex = self.vertices[row]
        deformerName = self.driverNames[column]
        theVtx = "{0}.vtx[{1}]".format(self.deformedShape, vertexIndex)
        print self.theSkinCluster, theVtx, deformerName, value
        # cmds.skinPercent( self.theSkinCluster,theVtx, transformValue=(deformerName, float (value)), normalize = True)
