#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
from dafoam import PYDAFOAM
import os
import numpy as np
from pyofm import PYOFM

gcomm = MPI.COMM_WORLD

os.chdir("./NACA0012_IRK")
if gcomm.rank == 0:
    os.system("rm -rf processor* *.bin")

# aero setup
U0 = 1.0

daOptions = {
    "solverName": "DAPimpleFoam",
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()

# read the U field at  2 and verify its norm
nLocalCells = DASolver.solver.getNLocalCells()
U = np.zeros(nLocalCells * 3)
ofm = PYOFM(gcomm)
ofm.readField("U", "volVectorField", "2", U)
UNorm = np.linalg.norm(U)
UNorm = gcomm.allreduce(UNorm, op=MPI.SUM)
print("UNorm", UNorm)

if abs(56.45964200732875 - UNorm) / 56.45964200732875 > 1e-6:
    print("DAPimpleDyMFoam test failed!")
    exit(1)
else:
    print("DAPimpleDyMFoam test passed!")
