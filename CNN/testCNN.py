import sys
import os
from transcription import *
from util_CNN import *

#warning are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

root_dir = os.getcwd()
res_dir = os.path.join(root_dir,"result")
test_path = sys.argv[1] if len(sys.argv) > 1 else ""
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
vcf_dir = sys.argv[3]

#créer le dossier ou on va stocker les données de performance de notre réseau
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

trLabel,predLabel,dico = testModel(test_path,BATCH_SIZE)
evaluateVariant(trLabel,predLabel)
for k in dico.keys():
    print(k,dico[k])
result_tab(dico,vcf_dir)