import splitfolders
import sys

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test

def splitFolder(input_folder,output_folder):
    """
    Partage les images de input_folder en trois sous dossiers : train(80%), test(10%) et val(10%)
    Entrée : Dossier qui contient un unique sous dossier nommé "images" dans lequel toutes les images du dataset sont stockées
    Sortie : input_folder + "Divided"
    """
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) 

def main(arg1):
    input_folder = arg1
    output_folder = input_folder + "Divided"
    splitFolder(arg1,output_folder)

if __name__ == "__main__":
    main(sys.argv[1])