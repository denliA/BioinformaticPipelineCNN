import sys
import os
from util_CNN import *
sys.path.append("/Neat/anaconda/envs/py/lib/python3.6/site-packages")
from rename import renameFiles
from splitData import splitFolder

#warning are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_folder = sys.argv[1]
if input_folder.endswith("/images"):
   input_folder = input_folder[:-len("/images")]
output_folder = input_folder + "Divided"
root_dir = os.getcwd()
train_dir = os.path.join(root_dir, output_folder + "/train/images")
valid_dir = os.path.join(root_dir, output_folder + "/val/images")
test_dir = os.path.join(root_dir, output_folder + "/test/images")

print(train_dir)


#enlever le "_paternal" et "_maternal" des noms des images
renameFiles(input_folder+"/images")

#partage le data en 3 dossiers train (80%), val(10%) et test(10%)
#le folder input doit contenir un dossier qui contient toutes les images
splitFolder(input_folder,output_folder)

#CNN
num_samplesTrain = len(getSamples(train_dir))
num_samplesValid = len(getSamples(valid_dir))

print("nb images d'entrainement : ",num_samplesTrain)
print("nb images de validation : ",num_samplesValid)

# creating TFRecords output folder
if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  

convertDataToTfrecord(train_dir,"train")
convertDataToTfrecord(valid_dir,"valid")

#Retourne une liste de fichiers qui match le pattern donné en paramètres
train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/train.tfrec")
valid_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/valid.tfrec")

#gradient descent is an iterative learning algorithm that uses a training dataset to update the weights a model. It minimizes the loss (difference between predicted label and true label)
#The batch size is a hyperparameter of gradient descent that controls the number of training samples to work through before the model’s internal parameters are updated.
#The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset

#si les arguments n'ont pas été précisées, on met la valeur par défaut dans le else
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 32 #nombre de images données au CNN en 1 seule fois
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 10 # nombre de cycles durant lesquelles on va entrainer le CNN avec toutes les données
print("batch_size : ",BATCH_SIZE)
print("epochs : ",EPOCHS)
print()

#get your datatensors for training
imageTrain, labelTrain = create_dataset(train_filenames,BATCH_SIZE)
imageTrain = tf.image.resize(imageTrain, size=(299, 299))
imageTrain = tf.image.convert_image_dtype(imageTrain, tf.float32)
imageTrain = tf.keras.applications.inception_v3.preprocess_input(imageTrain)

#get your datatensors for validation
imageValid, labelValid = create_dataset(valid_filenames,BATCH_SIZE)
imageValid = tf.image.resize(imageValid, size=(299, 299))
imageValid = tf.image.convert_image_dtype(imageValid, tf.float32)
imageValid = tf.keras.applications.inception_v3.preprocess_input(imageValid)

#get model
model = getModel(imageTrain,labelTrain)

#l'entrainement s'arrete plus tot si le loss atteint un pallier et ne diminue plus
#évite l'overfitting
early = tf.keras.callbacks.EarlyStopping( patience=10,
                                          min_delta=0.001,
                                          restore_best_weights=True)

#train model
history = model.fit(
    x=imageTrain,
    y=labelTrain,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[early],
    validation_data=(imageValid,labelValid),
    shuffle=False,
)

#afficher les performances de notre réseau
plotAccuracy(history)
plotLoss(history)

#évaluer notre modèle
score = model.evaluate(imageValid, labelValid, verbose=0)
print('\nValidation loss:', score[0])
print('Validation accuracy:', score[1])

#sauvegarder dans un dossier, les weights suite à l'entrainement de notre modèle
model.save("my_model")

