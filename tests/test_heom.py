# coding: utf-8
from typing import Tuple
from mugnier.libs.backend import array
from mugnier.structure.frame import MultiLayerMCTDH, Singleton, TensorTrain
from mugnier.structure.network import Node, End, Point, State
from mugnier.structure.operator import SumProdOp
from mugnier.heom.hierachy import ExtendedDensityTensor, Hierachy
