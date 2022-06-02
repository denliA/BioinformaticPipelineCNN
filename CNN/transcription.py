import csv
import os
from re import S
#Détection à partir de quelle ligne lire 
def capture(file) :
    """
    ROLE : Permet de savoir à partir de quelle ligne on doit lire
    ENTREE : file : path du file
    SORTIE : Liste de liste contenant les informations du VCF
    """
    #Permet à la boucle de savoir à partir de quelle ligne il doit stocker les valeurs
    t = False
    Stockage = []
    
    fichier = open(file,"r",encoding='ISO-8859–1')
    line = fichier.readline()
    while line :
        contenu = line.strip().split("\t")
        if t :
            Stockage.append(contenu)
        
        #On arrive à la ligne où la prochaine ligne contiendra le résultat des chromosomes donc on stocke les informations à ce moment
        if(contenu[0] == "#CHROM") :
            t = True
        line = fichier.readline()
    fichier.close()
    return Stockage


          
def write_csv(dataf) :
    """
    ROLE : Transcription du résultat du vcf sous forme d'un tableau csv
    ENTREE : fichier VCF
    SORTIE : NONE  
    """
    print("Un programme qui utilise csv.writer() pour écrire dans un fichier")
    print("\n")
    # Les données que nous allons écrire
    data = capture(dataf)
    # Ouvrir le fichier en mode écriture
    fichier = open('tableaux.csv','w')
    print("Nous avons ouvert le fichier tableaux, s'il n'existe pas il sera créé ")
    # Créer l'objet fichier
    obj = csv.writer(fichier)
    # Chaque élément de data correspond à une ligne
    for i in range(len(data)-1) :
        l = data[i]
        obj.writerow((l[1],l[2],l[3],l[4]))
    fichier.close()


def write_tsv(dataf) :
    """
    ROLE : Transcription du résultat du vcf sous forme d'un tableau tsv
    ENTREE : fichier VCF
    SORTIE : NONE  
    """
    data = capture(dataf)
    with open("tableau.tsv","wt") as out :
        tsv_writer = csv.writer(out, delimiter='\t')
        tsv_writer.writerow(["POS","ID","REF","ALT"])
        print("Création d'un tableau")
        for i in range(len(data)-1) :
            l = data[i]
            tsv_writer.writerow([l[1],l[2],l[3],l[4]])


def verif_pos(position,filevcf) :
    """
    Vérifie si la position du chromosome demandé se trouve dans le fichier vcf en entrée
    
    Args:
        position (string): Position du chromosome
        filevcf (string): Fichier du vcf

    Returns:
        List : Retourne le résultat du vcf à cette position du chromosome sous forme de list de list String
    """
    data = capture(filevcf)
    for d in data :
        if d[1] == position :
            return d
    return []              

def result_tab(dico,outputvcf) :
    """
    ROLE : Prend le résultat envoyé par le CNN et le résultat du vcf le met dans un tableau tsv
    ENTREE : dict => résultat CNN
             outputvcf => résultat outputvcf
    SORTIE : RIEN / Ecriture d'un tableau tsv
    """
    trouve = False
    #Création du tableau
    with open("tableau1.tsv","wt") as out :
        
        #Création des colonnes pour le tableaux
        tsv_writer = csv.writer(out, delimiter='\t')
        tsv_writer.writerow(["POS","REF","ALT","RESULTAT","POS-VCF","REF-VCF","ALT-VCF","CHEMIN DU FICHIER"])
        
        
        #Parcours le dictionnaire contenant le nom de l'image et le résultat renvoyé par le CNN
        for cle,valeur in dico.items() :
            
            #Parcours le dossier contenant les vcf pour correspondre le résultat du CNN à celui du VCF
            for fm in os.listdir(outputvcf):
                fm = os.path.join(outputvcf,fm)
                f = cle.split("_")
                if f[0] == "/pileup" :
                    pos = f[1].split(":")
                    temp = int(pos[1])+1
                    temp1 = str(temp)
                    data = verif_pos(temp1,fm)
                    if data != []:
                        trouve = True
                        print(pos[1],'est dans le fichier :',fm)
                        tsv_writer.writerow([temp1,f[2],f[3],valeur,data[1],data[3],data[4],fm])
        if (trouve == False):
            print("Les informations n'ont été trouvées dans aucun vcf, le fichier tableau1.tsv sera vide. \nVérifiez que vous avez bien copié le vcf de Deepvariant.")               



        



