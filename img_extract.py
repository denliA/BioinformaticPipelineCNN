#!/usr/bin/env python
import sys
sys.path.append("/Neat/anaconda/envs/py/lib/python3.6/site-packages")
from img_util import *


######variables/constantes######
example_path = ""
LABELS_CONST = {"RB":"(read_base)","BQ":"(base_quality)","MQ":"(mapping_quality)","S":"(strand)","RSV":"(read_support variant)","BDFR":"(base_differs_from ref)"}
liste_channels = list(LABELS_CONST.values())

if (len(sys.argv) != 3):
    exit()
    #si le nombre d'arguments n'est pas respecté.

TF_path = str(sys.argv[1])
RES_path = remove_slash(str(sys.argv[2])) + "/dv_images/images"

print(TF_path," ",RES_path)
img_extract(TF_path,RES_path,False)
