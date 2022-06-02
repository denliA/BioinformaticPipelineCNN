import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input,Flatten,Dense,Dropout


root_dir = os.getcwd()
#dossier qui contiendra les images qui résument la performance de notre CNN
res_dir = os.path.join(root_dir,"result")
#dossier qui contiendra nos tfrecords crées
tfrecords_dir = "tfrecords"

#créer le dossier s'il n'existe pas déjà
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


#nombre de labels
NUM_CLASSES=3
#notre réseau classe les images dans ces labels
class_names = ["deletion","insertion","changement_base"]
AUTOTUNE = tf.data.AUTOTUNE

def typeEvent(chaine):
    """
    Donne le type d'evenement genomique
    Entrée: Nom de fichier
    Retourne: Entier qui représente le type d'evenement genomique (basique)
    """
    split = chaine.split("_")[2],chaine.split("_")[3]
    #print(split[0],split[1])
    if(len(split[0]) > len(split[1])):
        return 0 #deletion
    elif(len(split[0]) < len(split[1])):
        return 1 #insertion
    else:
        return 2 #changement_base

#parser les metadonnées d'une image
def parseImage(path):
    """
    Donne les infos associées à une image
    Entrée: Chemin de l'image
    Retourne: Dictionnaire qui contient les données de vérité trouvées dans le titre de l'image
    """
    var = typeEvent(path)
    t = path.split("_")
    metadata = {"path":path,"locus":t[1],"ref":t[2],"alt":t[3],"variant":var}
    return metadata

def getSamples(directory):
    """
    Donne une liste de données associées aux images du répertoire
    Entrée: Dossier qui contient les images
    Retourne: liste de dictionnaires
    """
    samples = []
    with os.scandir(directory) as i:
        for entry in i:
            if entry.is_file() and entry.name.endswith(".png"):
                #print(entry.name)
                samples.append(parseImage(entry.name))
    return samples

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(image, path, example):
    """
    Chaque échantillon de données est appelé Example
    Entrée: image, chemin de l'image et les données associées à l'image (vérité récupérée via le titre de l'image)
    Retourne: Example cad le dictionnaire qui stock le mapping entre une clé et nos données
    """
    create_example.counter += 1
    feature = {
        "id":int64_feature(create_example.counter),
        "image": image_feature(image),
        "path": bytes_feature(path),
        "locus":bytes_feature(example["locus"]),
        "ref":bytes_feature(example["ref"]),
        "alt":bytes_feature(example["alt"]),
        "variant":int64_feature(example["variant"])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#initialiser le compteur qui associe un identifiant unique à chaque image
create_example.counter = 0

def parse_tfrecord_fn(example):
    """
    Décode le tfrecord qui était un fichier binaire
    Entrée: un Example de tfrecord
    Retourne: liste composée d'une image et du label qui lui est associé (le vrai variant obtenu grace au titre de l'image)
    """
    feature_description = {
        "id":tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "locus":tf.io.FixedLenFeature([], tf.string),
        "ref":tf.io.FixedLenFeature([], tf.string),
        "alt":tf.io.FixedLenFeature([], tf.string),
        "variant":tf.io.FixedLenFeature([], tf.int64)
    }
    sample = tf.io.parse_single_example(example, feature_description)
    picture = tf.io.decode_png(sample["image"], channels=3)
    label = tf.cast(sample["variant"], tf.float32)
    label = sample["variant"]
    return (picture,label)

def convertDataToTfrecord(path_dir,name):
    """
    Convertit les images contenues dans le dosseir path_dir en tfrecord
    Entrée : dossier qui contient toutes les images
    Sortie : nom du fichier .tfrecord
    """
    samples = getSamples(path_dir)
    with tf.io.TFRecordWriter(
        tfrecords_dir + "/" + name + ".tfrec"
    ) as writer:
        for sample in samples:
            image_path = path_dir + "/"+ sample["path"]
            image = tf.io.decode_png(tf.io.read_file(image_path))
            example = create_example(image, image_path, sample)
            writer.write(example.SerializeToString())

def seeTfrecord(enc_tfrecord):
    """
    Décode un tfrecord et affiche l'image et le label qui est devenu un tensor
    """
    raw_dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/{enc_tfrecord}")
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    for features in parsed_dataset.take(35):
        print(features[1])
        plt.figure(figsize=(7, 7))
        plt.imshow(features[0].numpy())
        plt.show()

def writeDecode(enc_tfrecord):
    """
    Décode un tfrecord et écrit dans un fichier texte
    Entrée: nom du fichier qui se finit par .tfrecord. Attention le chemin complet n'est pas inclus
    """
    dec_tfrecord = enc_tfrecord.replace(".tfrec",".text")
    f = open(f"{tfrecords_dir}/decoded_{dec_tfrecord}", "a")
    dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/{enc_tfrecord}")
    for d in dataset.take(35):
        example = tf.train.Example()
        example.ParseFromString(d.numpy())    
        f.write(str(example))
    f.close()

def readDecode(enc_tfrecord):
    """
    Décode un tfrecord et l'affiche sous forme de texte en clair. Les images sont encodés
    Entrée: nom du fichier qui se finit par .tfrecord. Attention le chemin complet n'est pas inclus
    """
    dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/{enc_tfrecord}")
    for d in dataset.take(35):
        example = tf.train.Example()
        example.ParseFromString(d.numpy())
        print(str(example))

def create_dataset(filenames, batch_size):
    """
    Donne un dataset
    Entrée: nom des fichiers tfrecord qui contiennent le dataset serialisé
    Retourne: liste qui contient une liste de images et une liste de labels qui lui sont associés
    """
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(batch_size * 10)
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    dataset.prefetch(AUTOTUNE)
    
    # Create an iterator
    iterator = iter(dataset)
    
    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    #image = tf.reshape(image, [1,299, 299, 3])
   
    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)   
    
    return (image, label)

def getModel(image,label):
    """
    Construit le modèle en utilisant le transfer learning du modèle préentrainé InceptionV3.
    On enlève les dernières couches de classification de l'ancien modèle.
    On lui rajoute nos couches qui correspondent à nos classes [deletion,insertion,changement_base].
    C'est la sortie de notre réseau de neurones.
    Entrée: liste de listes des images et des labels qui lui sont associées
    Retourne: notre modèle
    """
    #Combine it with keras
    model_input = Input(tensor=image,shape=(299,299))

    InceptionV3_model = InceptionV3(input_tensor=model_input, weights='imagenet', include_top=False)
    for layer in InceptionV3_model.layers:
        layer.trainable=False

    InceptionV3_last_output = InceptionV3_model.output
    InceptionV3_maxpooled_output = Flatten()(InceptionV3_last_output)
    InceptionV3_x = Dense(1024, activation='relu')(InceptionV3_maxpooled_output)
    InceptionV3_x = Dropout(0.5)(InceptionV3_x)
    InceptionV3_x = Dense(3, activation='softmax')(InceptionV3_x)
    InceptionV3_x_final_model = tf.keras.Model(inputs=InceptionV3_model.input,outputs=InceptionV3_x)

    #InceptionV3_x_final_model.summary()
    
    InceptionV3_x_final_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['categorical_accuracy']
    )
    return InceptionV3_x_final_model

def testModel(testing_dir,BATCH_SIZE):
    """
    Teste notre modèle après son entrainement sur des images contenues dans le dossier passé en paramètres
    Entrées : 
    Dossier qui contient les nouvelles images sur lequel le réseau ne s'est pas entrainé
    Taille du batch cad le nombre de images données au CNN en 1 seule fois
    Sortie : Liste de listes de classes prédites par le CNN et des vraies classes pour toutes les images
    du dossier et un dictionnaire ou clé=nom de l'image et valeur=variant
    """
    #Reconstruct the model from the saved weights
    reconstructed_model = keras.models.load_model("my_model")
    
    #get all the images in the validation folder
    test_samples = []
    with os.scandir(testing_dir) as i:
        for entry in i:
            if entry.is_file() and entry.name.endswith(".png"):
                test_samples.append(os.path.join(testing_dir,entry.name))
    
    # predict images
    #print("truth \t\t|\t prediction")
    truthLabel = []
    predictLabel = []
    dicoTest = {}
    for path in test_samples:
        img = image.load_img(path, target_size=(299,299))
        x_pred = np.array(img)
        tf.shape(x_pred)
        #imgplot = plt.imshow(x_pred)
        #plt.show()
        x_pred = np.expand_dims(x_pred, axis=0)
        x_pred = tf.image.convert_image_dtype(x_pred, tf.float32)
        x_pred = tf.keras.applications.inception_v3.preprocess_input(x_pred)
        images = np.vstack([x_pred])
        y_pred = reconstructed_model.predict(images, batch_size=BATCH_SIZE)
        picture_name = str(path).replace(str(testing_dir),"")
        truthLabel.append(class_names[typeEvent(picture_name)])
        predictLabel.append(class_names[np.argmax(y_pred, axis=-1)[0]])
        #print(class_names[typeEvent(picture_name)],"|",class_names[np.argmax(y_pred, axis=-1)[0]],sep='\t')
        dicoTest[picture_name]=class_names[np.argmax(y_pred, axis=-1)[0]]
    return (truthLabel,predictLabel,dicoTest)

def confusionMatrixDisplay(trueList,predictedList):
    """
    Dessine une matrice de confusion avec les noms des classes entre la verité et les prédictions
    Entrée : liste des labels predits et vrais
    """
    ConfusionMatrixDisplay.from_predictions(trueList, predictedList)
    plt.savefig(res_dir + "/ConfusionMatrix.png", dpi=300, bbox_inches='tight')
    #plt.show()

def accuracyMatrix(confusion_matrix):
    """
    Donne la précision cad le nombre de cas correctement trouvés
    Entrée : matrice de confusion
    Retourne : matrice de confusion entre les vrais labels et les classes prédites par le CNN
    """
    true_prediction = sum(np.diagonal(confusion_matrix))
    all_prediction = sum(confusion_matrix.flatten())
    print(true_prediction,all_prediction)
    accuracy = true_prediction/all_prediction
    ch1 = f"variants trouvés : {true_prediction}"
    ch2 = f"variants réels : {all_prediction}"
    print(ch1)
    print(ch2)
    f = open(res_dir + "/test.txt", "w")
    f.write("\n" + ch1)
    f.write("\n" + ch2)
    f.close()
    return accuracy

def evaluateVariant(trueLabel,predictedLabel):
    """
    Affiche la précision et la matrice de confusion pour les variants
    Entrée : liste des labels predits et de vrais labels
    """
    matrix = confusion_matrix(trueLabel,predictedLabel)
    accuracy = accuracyMatrix(matrix)
    print("*********** Test ***********")
    ch = f"précision : {accuracy}"
    print(ch)
    confusionMatrixDisplay(trueLabel,predictedLabel)
    print()

    f = open(res_dir + "/test.txt", "a")
    f.write("\n" + ch)
    f.close()

def plotAccuracy(history):
    """
    Trace un graphe qui mesure la performance du CNN :
    Accuracy en fonction du nombre d'epochs
    Entrée : retour de model.fit de keras qui lance l'entrainement
    """
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(res_dir + "/CNN_Accuracy.png", dpi=300, bbox_inches='tight')
    #plt.show()

def plotLoss(history):
    """
    Trace un graphe qui mesure la performance du CNN :
    Loss en fonction du nombre d'epochs
    Entrée : retour de model.fit de keras qui lance l'entrainement
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(res_dir + "/CNN_Loss.png", dpi=300, bbox_inches='tight')
    #plt.show()
