from __future__ import absolute_import
import os

WEIGHT_EDITOR = None
WEIGHT_EDITOR_ROOT = None

def runMWeightEditor():
    from .utils import rootWindow
    from .weightEditorWidget import SkinWeightWin

    # Keep global references around, otherwise they get GC'd
    global WEIGHT_EDITOR
    global WEIGHT_EDITOR_ROOT

    # make and show the UI
    if WEIGHT_EDITOR_ROOT is None:
        WEIGHT_EDITOR_ROOT = rootWindow()
    WEIGHT_EDITOR = SkinWeightWin(parent=WEIGHT_EDITOR_ROOT)
    WEIGHT_EDITOR.show()


if __name__ == "__main__":
    import sys

    folder = os.path.dirname(os.path.dirname(__file__))
    if folder not in sys.path:
        sys.path.insert(0, folder)
    runMWeightEditor()
