from __future__ import print_function
from __future__ import absolute_import
from maya import OpenMaya, OpenMayaAnim
from maya import cmds

from ctypes import c_double

import numpy as np
import re
from .utils import GlobalContext, getThreeIndices

from .abstractData import DataAbstract
from six.moves import range


# SKIN FUNCTIONS
class DataOfSkin(DataAbstract):
    def __init__(
        self,
        useShortestNames=False,
        hideZeroColumn=True,
        createDisplayLocator=True,
        mainWindow=None,
    ):
        self.useShortestNames = useShortestNames
        self.hideZeroColumn = hideZeroColumn
        self.clearData()
        super(DataOfSkin, self).__init__(
            createDisplayLocator=createDisplayLocator, mainWindow=mainWindow
        )
        self.isSkinData = True

    # MObject base function
    def getVerticesOrigShape(self, outPut=False):
        geometries = OpenMaya.MObjectArray()
        if outPut:
            self.sknFn.getOutputGeometry(geometries)
        else:
            self.sknFn.getInputGeometry(geometries)
        theMObject = geometries[0]

        return self.getVerticesShape(theMObject)

    # functions
    def smoothSkin(self, selectedIndices, repeat=1, percentMvt=1):
        rowsSel = []
        for item in selectedIndices:
            rowsSel += list(range(item[0], item[1] + 1))
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

    def swapOneOnOne(self, indicesSources, indicesDest):
        indicesSources = list(indicesSources)
        indicesDest = list(indicesDest)
        print(indicesSources, indicesDest)
        with GlobalContext(message="swapOneOnOne", doPrint=True):
            new2dArray = np.copy(self.orig2dArray)
            new2dArray[:, indicesDest] = (
                self.orig2dArray[:, indicesDest] + self.orig2dArray[:, indicesSources]
            )
            new2dArray[:, indicesSources] = 0
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

    def normalize(self):
        with GlobalContext(message="normalize", doPrint=True):
            new2dArray = np.copy(self.orig2dArray)
            unLock = np.ma.array(new2dArray.copy(), mask=self.lockedMask, fill_value=0)
            unLock.clip(0, 1)
            sum_unLock = unLock.sum(axis=1)
            unLockNormalized = (
                unLock
                / sum_unLock[:, np.newaxis]
                * self.toNormalizeToSum[:, np.newaxis]
            )

            np.copyto(new2dArray, unLockNormalized, where=~self.lockedMask)
            # normalize
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
        # normalize
        sum_unLock = unLock.sum(axis=1)
        unLockNormalized = (
            unLock / sum_unLock[:, np.newaxis] * arraySumNormalize[:, np.newaxis]
        )

        np.copyto(theArray, unLockNormalized, where=~theMask)

    def pruneWeights(self, pruneValue):
        with GlobalContext(message="pruneWeights", doPrint=True):
            new2dArray = np.copy(self.orig2dArray)
            self.pruneOnArray(
                new2dArray, self.lockedMask, self.toNormalizeToSum, pruneValue
            )
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
        # useRealIndices is for when the array is sparse(some defomers have been deleted)
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

                if re.search(leftSearch, influence, re.IGNORECASE) is not None:
                    oppInfluence = influence.replace(leftReplace, rightReplace)
                    break
                elif re.search(rightSearch, influence, re.IGNORECASE) is not None:
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
        print(oppDriverNames)
        return driverNames_oppIndices

    def mirrorArray(self, direction, leftInfluence="*_L_*", rightInfluence="*_R_*"):
        prt = (
            cmds.listRelatives(self.deformedShape, path=-True, parent=True)[0]
            if not cmds.nodeType(self.deformedShape) == "transform"
            else self.deformedShape
        )
        for att in [
            "symmetricVertices",
            "rightVertices",
            "leftVertices",
            "centerVertices",
        ]:
            if not cmds.attributeQuery(att, node=prt, exists=True):
                return
        symmetricVertices = cmds.getAttr(prt + ".symmetricVertices")

        with GlobalContext(message="mirrorArray", doPrint=self.verbose):
            driverNames_oppIndices = self.getArrayOppInfluences(
                leftInfluence=leftInfluence, rightInfluence=rightInfluence
            )
            if not driverNames_oppIndices:
                return

            # skin setting
            componentType = OpenMaya.MFn.kMeshVertComponent
            fnComponent = OpenMaya.MFnSingleIndexedComponent()
            userComponents = fnComponent.create(componentType)
            symVerts = [int(symmetricVertices[vert]) for vert in self.vertices]
            symVertsSorted = sorted(symVerts)
            indicesSort = [symVerts.index(vert) for vert in symVertsSorted]
            # vertices
            for vert in symVertsSorted:
                fnComponent.addElement(int(vert))
            # joints
            influenceIndices = OpenMaya.MIntArray()
            influenceIndices.setLength(self.nbDrivers)
            for i in range(self.nbDrivers):
                influenceIndices.set(i, i)
            # now the weights
            new2dArray = np.copy(self.display2dArray)
            new2dArray = new2dArray[:, np.array(driverNames_oppIndices)]
            new2dArray = new2dArray[np.array(indicesSort), :]
            self.softOn = False
            if new2dArray is not None:
                self.actuallySetValue(
                    new2dArray,
                    None,
                    userComponents,
                    influenceIndices,
                    self.shapePath,
                    self.sknFn,
                )
        if self.blurSkinNode and cmds.objExists(self.blurSkinNode):
            # set the vertices
            selVertices = self.orderMelList(symVertsSorted)
            inList = ["vtx[{0}]".format(el) for el in selVertices]
            cmds.setAttr(
                self.blurSkinNode + ".inputComponents",
                *([len(inList)] + inList),
                type="componentList"
            )

    def copyArray(self):
        self.copiedArray = np.copy(self.sub2DArrayToSet)
        self.copiedVerticesPosition = np.copy(self.getVerticesOrigShape())
        self.copiedVerticesIndices = self.vertices + []
        self.copiedColumnCount = self.columnCount
        self.copiedVertsPos = self.copiedVerticesPosition[self.Mtop: self.Mbottom + 1].copy()

    def pasteArray(self):
        if self.columnCount != self.copiedColumnCount:
            return False

        pasteVerticesPosition = np.copy(self.getVerticesOrigShape())
        pasteVertsPos = pasteVerticesPosition[self.Mtop: self.Mbottom + 1].copy()

        # make an array of distances
        a_min_b = pasteVertsPos[:, np.newaxis] - self.copiedVertsPos[np.newaxis, :]
        # compute length of the vectors
        distArray = np.linalg.norm(a_min_b, axis=2)
        # sort the indices of the closest
        sorted_columns_indices = distArray.argsort(axis=1)
        # now take the closest vertex
        closestIndex1 = sorted_columns_indices[:, 0]
        # get the array to place
        new2dArray = self.copiedArray[closestIndex1]
        self.actuallySetValue(
            new2dArray,
            self.sub2DArrayToSet,
            self.userComponents,
            self.influenceIndices,
            self.shapePath,
            self.sknFn,
        )

        return True

    def reassignLocally(self, reassignValue=1.0, nbJointsReassign=2):
        # print "reassignLocally"
        with GlobalContext(message="reassignLocally", doPrint=True):
            # 0 get orig shape
            self.origVerticesPosition = self.getVerticesOrigShape()
            self.origVertsPos = self.origVerticesPosition[self.Mtop: self.Mbottom + 1]

            # 2 get deformers origin position(bindMatrixInverse)
            depNode = OpenMaya.MFnDependencyNode(self.skinClusterObj)
            bindPreMatrixArrayPlug = depNode.findPlug("bindPreMatrix", True)
            lstDriverPrePosition = []

            for ind in self.indicesJoints:
                preMatrixPlug = bindPreMatrixArrayPlug.elementByLogicalIndex(ind)
                matFn = OpenMaya.MFnMatrixData(preMatrixPlug.asMObject())
                mat = matFn.matrix().inverse()
                position = [
                    OpenMaya.MScriptUtil.getDoubleArrayItem(mat[3], c) for c in range(3)
                ]
                lstDriverPrePosition.append(position)
            lstDriverPrePosition = np.array(lstDriverPrePosition)

            # 3 make an array of distances
            # compute the vectors from deformer to the point
            a_min_b = (
                self.origVertsPos[:, np.newaxis] - lstDriverPrePosition[np.newaxis, :]
            )
            # compute length of the vectors
            distArray = np.linalg.norm(a_min_b, axis=2)
            theMask = self.sumMasks
            distArrayMasked = np.ma.array(distArray, mask=~theMask, fill_value=0)
            # sort the columns
            sorted_columns_indices = distArrayMasked.argsort(axis=1)

            # now take the 2 closest columns indices(2 first) and do a dot product of the vectors
            closestIndices = sorted_columns_indices[:, :2]

            # make the vector of 2 closest joints
            closestIndex1 = sorted_columns_indices[:, 0]
            closestIndex2 = sorted_columns_indices[:, 1]
            closestDriversVector = (
                lstDriverPrePosition[closestIndex2]
                - lstDriverPrePosition[closestIndex1]
            )

            # now get the closest vectors to the point
            closestVectors_1 = a_min_b[
                np.arange(closestIndices.shape[0])[:], closestIndex1
            ]

            # now dot product
            A = closestVectors_1
            B = closestDriversVector
            resDot = np.einsum("ij, ij->i", A, B)  # A.B
            lengthVectorB = np.linalg.norm(B, axis=1)

            # we normalize, then  clip for the negative and substract from 1 to reverse the setting
            normalizedDotProduct = resDot / (lengthVectorB * lengthVectorB)
            normalizedDotProduct = normalizedDotProduct.clip(min=0.0, max=1.0)
            ################################################################################################################
            # FINISH #######################################################################################################
            ################################################################################################################
            theMask = self.sumMasks
            # now set the values in the array correct if cross product is positive
            addValues = np.full(self.orig2dArray.shape, 0.0)
            addValues[
                np.arange(closestIndex2.shape[0])[:], closestIndex2
            ] = normalizedDotProduct
            addValues[np.arange(closestIndex1.shape[0])[:], closestIndex1] = (
                1.0 - normalizedDotProduct
            )
            addValues = np.ma.array(addValues, mask=~theMask, fill_value=0)

            # 4 normalize it
            addValuesNormalized = addValues * self.toNormalizeToSum[:, np.newaxis]

            # 5 copy values
            new2dArray = np.copy(self.orig2dArray)
            np.copyto(new2dArray, addValuesNormalized, where=theMask)

            # 6 zero rest array
            ZeroVals = np.full(self.orig2dArray.shape, 0.0)
            np.copyto(new2dArray, ZeroVals, where=self.rmMasks)

            # the multiply value
            if reassignValue != 1.0:
                new2dArray = new2dArray * reassignValue + self.orig2dArray * (
                    1.0 - reassignValue
                )
            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
        # set Value
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

            # remaining array
            remainingArr = np.copy(self.orig2dArray)
            remainingData = np.ma.array(remainingArr, mask=~self.rmMasks, fill_value=0)
            sum_remainingData = remainingData.sum(axis=1)

            # first make new mask where remaining values are zero(so no operation can be done ....)
            zeroRemainingIndices = np.flatnonzero(sum_remainingData == 0)
            sumMasksUpdate = self.sumMasks.copy()
            sumMasksUpdate[zeroRemainingIndices, :] = False

            # add the values
            theMask = sumMasksUpdate if val == 0.0 else self.sumMasks
            absValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)

            # normalize the sum to the max value unLocked
            sum_absValues = absValues.sum(axis=1)
            absValuesNormalized = (
                absValues
                / sum_absValues[:, np.newaxis]
                * self.toNormalizeToSum[:, np.newaxis]
            )
            np.copyto(
                absValues,
                absValuesNormalized,
                where=sum_absValues[:, np.newaxis] > self.toNormalizeToSum[:, np.newaxis],
            )
            if val != 0.0:  # normalize where rest is zero
                np.copyto(
                    absValues,
                    absValuesNormalized,
                    where=sum_remainingData[:, np.newaxis] == 0.0,
                )
            sum_absValues = absValues.sum(axis=1)
            # non selected not locked Rest
            restVals = self.toNormalizeToSum - sum_absValues
            toMult = restVals / sum_remainingData
            remainingValues = remainingData * toMult[:, np.newaxis]
            # clip it
            remainingValues = remainingValues.clip(min=0.0, max=1.0)
            # renormalize

            # add with the mask
            np.copyto(new2dArray, absValues, where=~absValues.mask)
            np.copyto(new2dArray, remainingValues, where=~remainingValues.mask)

            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
        # set Value
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

            if average:
                if self.verbose:
                    print("average")
                theMask = sumMasksUpdate
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)
                sumCols_addValues = addValues.mean(axis=0)
                theTiling = np.tile(sumCols_addValues, (addValues.shape[0], 1))
                addValues = np.ma.array(theTiling, mask=~theMask, fill_value=0)
                addValues = val * addValues + (1.0 - val) * np.ma.array(
                    selectArr, mask=~theMask, fill_value=0
                )
            elif percent:  # percent Add
                addValues = np.ma.array(selectArr, mask=~theMask, fill_value=0)
                addValues = (1 + val) * addValues
            else:  # regular add
                valuesToAdd = val / self.nbIndicesSettable[:, np.newaxis]
                addValues = (
                    np.ma.array(selectArr, mask=~theMask, fill_value=0) + valuesToAdd
                )
            addValues = addValues.clip(min=0, max=1.0)
            # normalize the sum to the max value unLocked
            sum_addValues = addValues.sum(axis=1)
            addValuesNormalized = (
                addValues
                / sum_addValues[:, np.newaxis]
                * self.toNormalizeToSum[:, np.newaxis]
            )
            np.copyto(
                addValues,
                addValuesNormalized,
                where=sum_addValues[:, np.newaxis]
                > self.toNormalizeToSum[:, np.newaxis],
            )
            # normalize where rest is zero
            np.copyto(
                addValues,
                addValuesNormalized,
                where=sum_remainingData[:, np.newaxis] == 0.0,
            )
            sum_addValues = addValues.sum(axis=1)

            # non selected not locked Rest
            restVals = self.toNormalizeToSum - sum_addValues
            toMult = restVals / sum_remainingData
            remainingValues = remainingData * toMult[:, np.newaxis]
            # clip it
            remainingValues = remainingValues.clip(min=0.0, max=1.0)
            # renormalize
            if autoPrune:
                self.pruneOnArray(
                    remainingValues,
                    remainingValues.mask,
                    remainingValues.sum(axis=1),
                    autoPruneValue,
                )
                self.pruneOnArray(
                    addValues, addValues.mask, addValues.sum(axis=1), autoPruneValue
                )
            # add with the mask
            np.copyto(new2dArray, addValues, where=~addValues.mask)
            np.copyto(new2dArray, remainingValues, where=~remainingValues.mask)
            if self.softOn:  # mult soft Value
                new2dArray = (
                    new2dArray * self.indicesWeights[:, np.newaxis]
                    + self.orig2dArray * (1.0 - self.indicesWeights)[:, np.newaxis]
                )
        # set Value
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
        self.undoDic["inListVertices"] = []
        # if connected
        if self.blurSkinNode:
            # set the vertices
            if self.indicesVertices.size > 0:
                selVertices = self.orderMelList(self.indicesVertices)
                inList = ["vtx[{0}]".format(el) for el in selVertices]
                self.undoDic["inListVertices"] = inList
                if cmds.objExists(self.blurSkinNode):
                    cmds.setAttr(
                        self.blurSkinNode + ".inputComponents",
                        *([len(inList)] + inList),
                        type="componentList"
                    )

    def actuallySetValue(
        self,
        new2dArray,
        sub2DArrayToSet,
        userComponents,
        influenceIndices,
        shapePath,
        sknFn,
    ):
        with GlobalContext(message="actuallySetValue", doPrint=self.verbose):
            if self.softOn:
                arrayForSetting = np.copy(new2dArray[self.subOpposite_sortedIndices])
            else:
                arrayForSetting = np.copy(new2dArray)
            with GlobalContext(message="OpenMaya setWeights", doPrint=self.verbose):
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

                # with GlobalContext(message = "sknFn.setWeights"):
                normalize = False
                UndoValues = OpenMaya.MDoubleArray()
                sknFn.setWeights(
                    shapePath,
                    userComponents,
                    influenceIndices,
                    newArray,
                    normalize,
                    UndoValues,
                )
            if self.storeUndo:
                self.undoValues = UndoValues
                self.storeUndo = False
            self.redoValues = newArray

            # do the stting in the 2dArray
            if sub2DArrayToSet is not None and sub2DArrayToSet.size != 0:
                np.put(sub2DArrayToSet, range(sub2DArrayToSet.size), new2dArray)
                self.computeSumArray()

    # get data
    def exposeSkinData(self, inputSkinCluster, indices=[], getskinWeights=True):
        self.skinClusterObj = self.getMObject(inputSkinCluster, returnDagPath=False)
        self.sknFn = OpenMayaAnim.MFnSkinCluster(self.skinClusterObj)

        jointPaths = OpenMaya.MDagPathArray()
        self.sknFn.influenceObjects(jointPaths)
        self.indicesJoints = []
        for i in range(jointPaths.length()):
            ind = self.sknFn.indexForInfluenceObject(jointPaths[i])
            self.indicesJoints.append(ind)
        geometries = OpenMaya.MObjectArray()
        self.sknFn.getOutputGeometry(geometries)
        self.shapePath = OpenMaya.MDagPath().getAPathTo(geometries[0])

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
        if self.shapePath.apiType() == OpenMaya.MFn.kNurbsCurve:
            componentType = OpenMaya.MFn.kCurveCVComponent
            crvFn = OpenMaya.MFnNurbsCurve(self.shapePath)
            vertexCount = crvFn.numCVs()
        elif self.shapePath.apiType() == OpenMaya.MFn.kNurbsSurface:
            self.isNurbsSurface = True
            componentAlreadyBuild = True
            componentType = OpenMaya.MFn.kSurfaceCVComponent
            MfnSurface = OpenMaya.MFnNurbsSurface(self.shapePath)
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

        weights = OpenMaya.MDoubleArray()
        if not getskinWeights:
            return weights
        intptrUtil = OpenMaya.MScriptUtil()
        intptrUtil.createFromInt(0)
        intPtr = intptrUtil.asUintPtr()

        self.sknFn.getWeights(self.shapePath, self.fullComponent, weights, intPtr)
        return weights

    def convertRawSkinToNumpyArray(self):
        # deadFast
        res = OpenMaya.MScriptUtil(self.rawSkinValues)
        ptr = res.asDoublePtr()

        lent = self.rawSkinValues.length()
        with GlobalContext(message="convertingSkinValues", doPrint=self.verbose):
            cta = (c_double * lent).from_address(int(ptr))
            arr = np.ctypeslib.as_array(cta)
            self.raw2dArray = np.copy(arr)
            self.raw2dArray = np.reshape(self.raw2dArray, (-1, self.nbDrivers))
        # reorder
        if self.softOn:  # order with indices
            self.display2dArray = self.raw2dArray[self.sortedIndices]
        else:
            self.display2dArray = self.raw2dArray
        # now find the zeroColumns

        myAny = np.any(self.raw2dArray, axis=0)
        self.usedDeformersIndices = np.where(myAny)[0]
        self.hideColumnIndices = np.where(~myAny)[0]
        self.computeSumArray()

    def rebuildRawSkin(self):
        if self.fullShapeIsUsed:
            self.rawSkinValues = self.exposeSkinData(self.theSkinCluster)
        else:
            self.rawSkinValues = self.exposeSkinData(
                self.theSkinCluster, indices=self.vertices
            )

    def getSkinClusterValues(self, skinCluster):
        driverNames = cmds.skinCluster(skinCluster, q=True, inf=True)
        skinningMethod = cmds.getAttr(skinCluster + ".skinningMethod")
        normalizeWeights = cmds.getAttr(skinCluster + ".normalizeWeights")
        return (driverNames, skinningMethod, normalizeWeights)

    def computeSumArray(self):
        if self.raw2dArray is not None:
            self.sumArray = self.raw2dArray.sum(axis=1)

    def getNamesHighestColumns(self):
        columnSum = -self.display2dArray.sum(axis=0)
        sorted_columns_indices = columnSum.argsort()
        return np.array(self.driverNames)[sorted_columns_indices].tolist()

    def getDriversShortNames(self):
        self.shortColumnsNames = []
        for el in self.driverNames:
            shortName = el.split(":")[-1].split("|")[-1]
            if self.useShortestNames and shortName.startswith("Dfm_"):
                splt = shortName.split("_")
                shortName = " ".join(splt[1:])
            self.shortColumnsNames.append(shortName)

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
                        cmds.disconnectAttr(
                            inConn[0], self.theSkinCluster + ".weightList"
                        )
                return self.blurSkinNode
        return ""

    # redefine abstract data functions
    def clearData(self):
        super(DataOfSkin, self).clearData()
        self.AllWght = []
        self.theSkinCluster = ""
        self.blurSkinNode = ""

        self.driverNames = []
        self.indicesJoints = []
        self.nbDrivers = 0
        self.skinningMethod = ""
        self.normalizeWeights = []

        self.undoDic = {"isSkin": True, "inListVertices": [], "theSkinCluster": ""}

    def getAllData(
        self,
        displayLocator=True,
        getskinWeights=True,
        force=True,
        inputVertices=None,
        **kwargs
    ):
        success = self.getDataFromSelection(
            typeOfDeformer="skinCluster",
            force=force,
            inputVertices=inputVertices,
            **kwargs
        )
        if not success or self.theDeformer == "":
            if not force:
                return False
        else:
            self.theSkinCluster = self.theDeformer
        # get skin infos vertices
        if (
            not self.theSkinCluster
            or not cmds.objExists(self.theSkinCluster)
            or not cmds.nodeType(self.theSkinCluster) == "skinCluster"
        ):
            return False
        (
            self.driverNames,
            self.skinningMethod,
            self.normalizeWeights,
        ) = self.getSkinClusterValues(self.theSkinCluster)
        self.getDriversShortNames()
        self.nbDrivers = len(self.driverNames)

        self.columnsNames = self.driverNames

        # use vertex selection 
        with GlobalContext(message="rawSkinValues", doPrint=self.verbose):
            self.getSoftSelectionVertices(inputVertices=inputVertices)

            if not self.vertices:
                self.vertices = cmds.getAttr(
                    "{0}.weightList".format(self.theSkinCluster), multiIndices=True
                )
                self.verticesWeight = [1.0] * len(self.vertices)
                self.sortedIndices = list(range(len(self.vertices)))
                self.opposite_sortedIndices = list(range(len(self.vertices)))
                self.softOn = 0
                self.fullShapeIsUsed = True
                self.rawSkinValues = self.exposeSkinData(
                    self.theSkinCluster, getskinWeights=getskinWeights
                )
            else:
                self.rawSkinValues = self.exposeSkinData(
                    self.theSkinCluster,
                    indices=self.vertices,
                    getskinWeights=getskinWeights,
                )
                self.fullShapeIsUsed = False
        if displayLocator:
            self.connectDisplayLocator()
        if getskinWeights:
            self.createRowText()
        self.hideColumnIndices = []
        self.usedDeformersIndices = list(range(self.nbDrivers))

        self.rowCount = len(self.vertices)
        self.columnCount = self.nbDrivers

        self.getLocksInfo()
        if getskinWeights:
            self.convertRawSkinToNumpyArray()
        return True

    def preSettingValuesFn(self, chunks, actualyVisibleColumns):
        # first check if connected
        self.getConnectedBlurskinDisplay(disconnectWeightList=True)

        super(DataOfSkin, self).preSettingValuesFn(chunks, actualyVisibleColumns)

        # get normalize values
        toNormalizeTo = np.ma.array(
            self.orig2dArray, mask=~self.lockedMask, fill_value=0.0
        )
        self.toNormalizeToSum = 1.0 - toNormalizeTo.sum(axis=1).filled(0.0)

        # NOW Prepare for settingSkin Cluster
        self.influenceIndices = OpenMaya.MIntArray()
        self.influenceIndices.setLength(self.nbDrivers)
        for i in range(self.nbDrivers):
            self.influenceIndices.set(i, i)
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
        lengthArray = self.nbDrivers * (self.Mbottom - self.Mtop + 1)
        self.newArray = OpenMaya.MDoubleArray()
        self.newArray.setLength(lengthArray)

        self.undoDic.update(
            {
                "theSkinCluster": self.theSkinCluster,
                "userComponents": self.userComponents,
                "influenceIndices": self.influenceIndices,
                "shapePath": self.shapePath,
                "sknFn": self.sknFn,
            }
        )
        # set normalize FALSE
        cmds.setAttr(self.theSkinCluster + ".normalizeWeights", 0)

    def getValue(self, row, column):
        return (
            self.display2dArray[row][column]
            if column < self.nbDrivers
            else self.sumArray[row]
        )

    def setValue(self, row, column, value):
        vertexIndex = self.vertices[row]
        deformerName = self.driverNames[column]
        theVtx = "{0}.vtx[{1}]".format(self.deformedShape, vertexIndex)
        if self.verbose:
            print(self.theSkinCluster, theVtx, deformerName, value)

    # locks
    def getLocksInfo(self):
        super(DataOfSkin, self).getLocksInfo()

        self.lockedColumns = []
        for driver in self.driverNames:
            isLocked = False
            if cmds.attributeQuery("lockInfluenceWeights", node=driver, exists=True):
                isLocked = cmds.getAttr(driver + ".lockInfluenceWeights")
            self.lockedColumns.append(isLocked)

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

    def lockRows(self, selectedIndices, doLock=True):
        super(DataOfSkin, self).lockRows(selectedIndices, doLock=doLock)
        if not self.blurSkinNode or not cmds.objExists(self.blurSkinNode):
            self.getConnectedBlurskinDisplay()
        if self.blurSkinNode and cmds.objExists(self.blurSkinNode):
            cmds.setAttr(self.blurSkinNode + ".getLockWeights", True)
            # update

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

    def selectDeformers(self, selectedIndices):
        toSel = [
            self.driverNames[column]
            for column in selectedIndices
            if cmds.objExists(self.driverNames[column])
        ]
        cmds.select(toSel)
        cmds.selectMode(object=True)

    # callBacks
    def renameCB(self, oldName, newName):
        if not cmds.objExists(newName):
            return
        if oldName in self.driverNames:
            ind = self.driverNames.index(oldName)
            self.driverNames[ind] = newName
            self.getDriversShortNames()
        if oldName == self.theSkinCluster:
            self.theSkinCluster = newName
        if oldName == self.deformedShape:
            self.deformedShape = newName
            prt = cmds.listRelatives(newName, p=True, path=True)[0]
            if prt and cmds.objExists(prt):
                shapeShortName = prt.split(":")[-1].split("|")[-1]
                splt = shapeShortName.split("_")
                if len(splt) > 5:
                    shapeShortName = "_".join(splt[-7:-4])
                self.shapeShortName = shapeShortName
