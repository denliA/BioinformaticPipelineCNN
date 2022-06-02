import sys
import os

def renameFiles(input_folder):
    """
    Change le nom des images dans input_folder pour enlever "_paternal" et "_maternal" qui vient de la
    génération d'images de DeepVariant mais qui ne correspond pas à notre format des titres de nos images
    Entrée : dossier qui contient des images pileup
    """
    count=0
    with os.scandir(input_folder) as i:
        for entry in i:
            if entry.is_file() and entry.name.endswith(".png"):
                old_name = os.path.join(input_folder, entry.name)
                #supprime le str qu'on veut pas dans le titre de l'image
                new_name = old_name.replace("_paternal","")
                if old_name!=new_name:
                    os.rename(old_name,new_name)
                    count +=1
                new_name = old_name.replace("_maternal","")
                if old_name!=new_name:
                    os.rename(old_name,new_name)
                    count +=1
    print("Fichiers renommés : ",count)

def main(arg1):
    renameFiles(arg1)


if __name__ == "__main__":
    main(sys.argv[1])
