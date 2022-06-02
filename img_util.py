from array import array
import tensorflow as tf
import os
from PIL import Image
import shutil
import sys
sys.path.append("/Neat/anaconda/envs/py/lib/python3.6/site-packages")
from nucleus.util import vis


def typeEvent(example):
  """Retourne le type d'evenement genomique
  Arguments:
    Un fichier exemple Deepvariant.
  Retourne:
    le type d'evenement genomique (basique)
  """
  chaine = vis.locus_id_with_alt(example)
  split = (chaine.split("_")[-2],chaine.split("_")[-1])
  
  if(len(split[0]) > len(split[1])):
    return "deletion"
  elif(len(split[0]) < len(split[1])):
    return "insertion"
  else:
    return "changement_base"


def remove_slash(chaine):
    """
    Enlève le slash
    Entrée : chaine de caractères qui est un chemin finissant par un slash
    Retourne : la chaine sans slash
    """
    if chaine[-1] == '\n':
        return chaine[:-1]
    return chaine

    #donne l'event ins ou del...
    """
    Donne le type de variant (insertion,deletion,changement_base)
    Entrée: fichier example
    Retourne: le type d'evenement genomique
    """
    chaine = vis.locus_id_with_alt(example)
    split = (chaine.split("_")[-2],chaine.split("_")[-1])
  
    if(len(split[0]) > len(split[1])):
        return "deletion"
    elif(len(split[0]) < len(split[1])):
        return "insertion"
    else:
        return "changement_base"

def images_dv_concat(filename_6,filename_rgb,hauteur,nom_final):
    """
    str x str -> img
    A partir du chemin de deux images concatène les 2 types de pileup images de DeepVariant (DV)
    Entrées :
    Les images brutes de DeepVariant
    Hauteur selon le total coverage de VARSIM
    """
    crop_rgb = Image.open(filename_rgb)
    crop_6 = Image.open(filename_6)
    width_rgb_og = crop_rgb.width
    width_6 = crop_6.width
    "récupération de la longueur des 2 images"

    croped_rgb = crop_rgb.crop((0,0,width_rgb_og,hauteur))
    croped_6 = crop_6.crop((0,0,width_6,hauteur))
    #hauteur selon total coverage de VARSIM.


    final_img = Image.new("RGB",(width_6+width_rgb_og,hauteur),"white")
    final_img.paste(croped_6,(0,0))
    final_img.paste(croped_rgb,(width_6,0))
    final_img.save(nom_final)

def img_extract(TF_path,RES_path,oui):
    """
    Crée des images à partir des tfrecords
    Entrées :
    Tfrecords
    Chemin du dossier des résultats
    Boolean

    Retourne :  chemin du dossier qui contient les images générés
    """
    dataset = tf.data.TFRecordDataset(TF_path,compression_type="GZIP")
    dataset_length = 1
    #Cherche a connaître le nombre de données dans le Tfrecord
    try:
        dataset_length = [i for i,_ in enumerate(dataset)][-1] + 1
        if oui:
            print(dataset_length)
    except:
        print("une erreur est survenue dans la fonction, les TFRecord sont surement vides")
    
    #Crée la succession de dossier données avec RES_path afin de contenir les résultats
    res_tmp = str(RES_path) + "/tmp"
    isdir = os.path.isdir(RES_path)
    isdir_tmp = os.path.isdir(res_tmp)

    if(isdir==False and isdir_tmp==False):
        try:
            os.makedirs(RES_path)
            os.makedirs(res_tmp)
            print("Création du dossier pour les images\n")
        except:
            print("UNE ERREUR DANS LA CREATION DU DOSSIER, il existe peut-être déjà ou que la chaine de caractères ne termine pas par / \nVérifiez également les droits d'accès.")
            exit()
    print(dataset_length)
    for e in dataset.take(dataset_length):
        example = tf.train.Example()
        example.ParseFromString(e.numpy())

        #cree les pileup images en deux version 1) les 6 channels concatenées 2) fusionnées en RGB
        filename = res_tmp + '/pileup_{}_truth={}.png'.format(vis.locus_id_with_alt(example), vis.label_from_example(example))
        filename_RGB = res_tmp + '/pileup_{}_RGB_truth={}.png'.format(vis.locus_id_with_alt(example), vis.label_from_example(example))
        vis.draw_deepvariant_pileup(example, path=filename, show=False)
        vis.draw_deepvariant_pileup(example, path=filename_RGB, show=False,composite_type="RGB",annotated=False)
        
        
        filename_final = RES_path + '/pileup_{}_full_truth={}.png'.format(vis.locus_id_with_alt(example), vis.label_from_example(example))
        images_dv_concat(filename,filename_RGB,50,filename_final)
        print(vis.locus_id_with_alt(example),typeEvent(example))
    isdir_tmp = os.path.isdir(res_tmp)