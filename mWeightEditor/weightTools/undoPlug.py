# https://medium.com/@k_serguei/maya-python-api-2-0-and-the-undo-stack-80b84de70551
from __future__ import absolute_import
import sys
import _ctypes
import maya.OpenMayaMPx as OpenMayaMPx
import six


class PythonCommand(OpenMayaMPx.MPxCommand):
    s_name = "pythonCommand"

    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    @staticmethod
    def creator():
        return OpenMayaMPx.asMPxPtr(PythonCommand())

    def doIt(self, args):
        ptr = six.integer_types[-1](args.asString(0), 0)
        self._imp = _ctypes.PyObj_FromPtr(ptr)

        # we could pass a *args and a **kwargs to have direct access to values
        self._imp.doIt()

    def redoIt(self):
        self._imp.redoIt()

    def undoIt(self):
        # print "pythonCommand - Undo"
        self._imp.undoIt()

    def isUndoable(self):
        return True


##############################################################################
#
# The following routines are used to register/unregister
# the command we are creating within Maya
#
##############################################################################
def initializePlugin(plugin):
    pluginFn = OpenMayaMPx.MFnPlugin(plugin)
    try:
        pluginFn.registerCommand(PythonCommand.s_name, PythonCommand.creator)
    except Exception:
        sys.stderr.write("Failed to register command: %s\n" % PythonCommand.s_name)
        raise


# Uninitialize the script plug-in
def uninitializePlugin(plugin):
    pluginFn = OpenMayaMPx.MFnPlugin(plugin)
    try:
        pluginFn.deregisterCommand(PythonCommand.s_name)
    except Exception:
        sys.stderr.write("Failed to unregister command: %s\n" % PythonCommand.s_name)
        raise
