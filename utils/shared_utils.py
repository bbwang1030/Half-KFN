# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from enum import Enum

# -------------------------------------------------
# ENUMS
# -------------------------------------------------

class TestDimensionality(Enum):
    One = 1
    Multi = 2
    Bin = 3


class OnedimensionalTest(Enum):
    KS = 1
    AD = 2


class MultidimensionalTest(Enum):
    MMD = 0
    FR = 1
    Energy = 2
    KNN = 3
    SmoothKNN = 4
    halfKFN = 5
    KFN_ours = 6
    KFN = 7
    halfKNN = 8

def MultidimensionalTest_name(name):
    if name == "halfKFN":
        name = "Half-KFN with permutation"
    elif name == "KFN_ours":
        name = "Half-KFN with bootstrap"
    else:
        name = name
    return name
        # MMD = "MMD"
        # FR = "FR"
        # Energy = "Energy"
        # KNN = "KNN"
        # SmoothKNN = "SmoothKNN"
        # halfKFN = "Half-KFN with permutation"
        #
        # KFN = 8
        # KFN_ours = "Half-KFN with bootstrap"
        # halfKNN = 7



class DimensionalityReduction(Enum):
    NoRed = 0
    PCA = 1
    SRP = 2
    UAE = 3
    TAE = 4
    BBSDs = 5
    BBSDh = 6
    Classif = 7
    simulation = 8
