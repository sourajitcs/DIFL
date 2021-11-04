import os
import sys
import time
import datetime
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc
import shutil
import glob
import models

gpu = 0

# helper function to print nice strings
def prettify_number(number):
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(number % 10, 4)]
    if 11 <= (number % 100) <= 13:
        suffix = 'th'
    return str(number) + suffix

# define the main training step
@tf.function
def training_step(classification_x_batch, classification_y_batch, domain_x_batch, domain_y_batch, domain_l):
    
    # classification training step
    with tf.GradientTape(persistent=True) as tape:
        logits_classification = overall_classification_model(classification_x_batch, training=True)
        classification_batch_loss = binary_loss(classification_y_batch, logits_classification)
    classification_gradients = tape.gradient(classification_batch_loss, overall_classification_model.trainable_weights)
 
    del tape
    
    # update classification accuracy and loss
    classification_accuracy.update_state(classification_y_batch, logits_classification)
    classification_loss.update_state(classification_batch_loss)
    
    # update the generator and classifier models
    classification_optimizer.apply_gradients(zip(classification_gradients, overall_classification_model.trainable_weights))

    # define the generator labels used for calculating the generator loss
    generator_labels = tf.fill([domain_l,1],0.5)  

    # GAN training step
    with tf.GradientTape(persistent=True) as tape:
        encodings_generator = generator_model(domain_x_batch, training=True)
        logits_discriminator = discriminator_model(encodings_generator, training=True)
        discriminator_batch_loss = binary_loss(domain_y_batch, logits_discriminator)
        generator_batch_loss = binary_loss(generator_labels, logits_discriminator)
    discriminator_gradients = tape.gradient(discriminator_batch_loss, discriminator_model.trainable_weights)
    generator_gradients = tape.gradient(generator_batch_loss, generator_model.trainable_weights)
   
    del tape
     
    # update the domain accuracy and other metrics
    domain_accuracy.update_state(domain_y_batch, logits_discriminator)
    discriminator_loss.update_state(discriminator_batch_loss)
    generator_loss.update_state(generator_batch_loss)
    
    # update the generator and discriminator models
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_model.trainable_weights))
    generator_optimizer.apply_gradients(zip(generator_gradients, generator_model.trainable_weights))

# define the testing step
@tf.function
def testing_step(x_batch, y_batch):
    final_logits = overall_classification_model(x_batch, training=False)
    classification_accuracy.update_state(y_batch, final_logits)

# parse the input directory
run_directory = sys.argv[1]
if run_directory.split("/")[-1] == "":
    root_dir = "runs/" + "/".join(run_directory.split("/")[1:-1]) + "/"
else:
    root_dir = "runs/" + "/".join(run_directory.split("/")[1:]) + "/"

# parse the saved model inputs
with open(root_dir + "hyperparameters.txt", "r") as f:
    start_time = f.readline().split(":")[-1][1:-1]
    line = f.readline().split(",")
    domain0_directory, domain1_directory = line[0].split(":")[-1][1:], line[1].split(":")[-1][1:-1]
    line = f.readline().split(":")[-1]
    img_height, img_width = int(line.split(",")[0][1:]), int(line.split(",")[1][1:-1])
    line = f.readline().split(":")[-1]
    secondary_img_height, secondary_img_width = int(line.split(",")[0][1:]), int(line.split(",")[1][1:-1])
    num_of_filters = int(f.readline().split(":")[-1][1:-1])
    batch_size = int(f.readline().split(":")[-1][1:-1])
    classification_lr = float(f.readline().split(":")[-1][1:-1])
    generator_lr = discriminator_lr = float(f.readline().split(":")[-1][1:-1])
    generator_weights = f.readline().split(":")[-1][1:-1]
    domain0_seed = int(f.readline().split(":")[-1][1:-1])
    domain1_seed = int(f.readline().split(":")[-1][1:-1])
    if generator_weights == "None":
        generator_weights = None
with open(root_dir + "saved/num_epochs.txt", "r") as f:
    line = f.readline().split(":")[-1]
    class0_count, class1_count = int(line.split(",")[0][1:]), int(line.split(",")[1][1:-1])
    line = f.readline().split(":")[-1]
    domain0_count, domain1_count = int(line.split(",")[0][1:]), int(line.split(",")[1][1:-1])
    epoch = int(f.readline().split(":")[-1][1:-1])
    batch = int(f.readline().split(":")[-1][1:-1])
    domain1_maximum_accuracy = float(f.readline().split(":")[-1][1:-1])
'''
# print confirmation screen
print(f"""
Testing the run in the {root_dir} folder!
Start Date & Time: {start_time}
Source Domain: {domain0_directory}, Target Domain: {domain1_directory}
Image Parameters: {img_height}, {img_width}
Secondary Image Parameters: {secondary_img_height}, {secondary_img_width}
Number of Filters: {num_of_filters}
Batch Size: {batch_size}
Classification Learning Rate: {classification_lr}
Generator & Discriminator Learning Rate: {generator_lr}
Type of Generator Weights: {generator_weights}
Domain 0 Seed: {domain0_seed}
Domain 1 Seed: {domain1_seed}
""")

if input("Continue?[Y/N]") == 'y':
    pass
else:
    exit()
'''
# parse domain names
domain0_name = "tbx"
domain1_name = domain1_directory.split("/")[-2].split("-")[0]

print(f"\nImporting the necessary datasets..")

# create the classification datasets
domain0_train_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain0_name}/train/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
domain0_train_dataset = domain0_train_dataset.cache()

# import and cache the testing datasets
domain0_test_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain0_name}/test/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
domain1_train_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain1_name}/train/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
domain1_test_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain1_name}/test/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
domain0_test_dataset = domain0_test_dataset.cache()
domain1_train_dataset = domain1_train_dataset.cache()
domain1_test_dataset = domain1_test_dataset.cache()

print("Done importing the necessary datasets!")


'''
Load the DIFL network models
'''
generator_model = keras.models.load_model(root_dir + "saved/generator_model")
discriminator_model = keras.models.load_model(root_dir + "saved/discriminator_model")
classifier_model = keras.models.load_model(root_dir + "saved/classifier_model")
overall_classification_model = models.create_overall_classification_model(img_height, img_width, generator_model, classifier_model)

'''
Specify the other parameters for the networks
'''

# instantiate the accuracy metrics
classification_accuracy = keras.metrics.BinaryAccuracy()

'''
Tensorboard initialization
'''
accuracy_writer = tf.summary.create_file_writer(root_dir + f"accuracy-logs/")

'''
Start training the model as well as its evaluation
'''
 
# conduct testing 

# test the model on 1st domain test
for i,(xbatch, ybatch) in enumerate(domain1_test_dataset):
    
    if i == len(domain1_test_dataset)-1:
        print(f"Testing on batch {i}...")

    elif i%3 == 0:
        print(f"Testing on batch {i}.", end='\r')
    elif i%3 == 1:
        print(f"Testing on batch {i}..", end='\r')
    else:
        print(f"Testing on batch {i}...", end='\r')
    
    testing_step(xbatch, ybatch)
domain1_test_accuracy = float(classification_accuracy.result())
classification_accuracy.reset_states()

print("Done testing on the Domain 1 Test set!")

# print the result
#print(f"The accuracy on the 1st domain training set is: {domain0_train_accuracy}.")
#print(f"The accuracy on the 1st domain testing set is: {domain0_test_accuracy}.")
#print(f"The accuracy on the 2nd domain training set is: {domain1_train_accuracy}.")
print(f"The accuracy on the 2nd domain testing set is: {domain1_test_accuracy}.")

# write results to tensorboard
with accuracy_writer.as_default():
    #tf.summary.scalar("0th Domain Train Accuracy", domain0_train_accuracy, step=epoch)
    #tf.summary.scalar("0th Domain Test Accuracy", domain0_test_accuracy, step=epoch)
    #tf.summary.scalar("1st Domain Train Accuracy", domain1_train_accuracy, step=epoch)
    tf.summary.scalar("1st Domain Test Accuracy", domain1_test_accuracy, step=epoch)

# check if current results are better than the past best results
if domain1_test_accuracy > domain1_maximum_accuracy:
    domain1_maximum_accuracy = domain1_test_accuracy
    print(f"Maximum Domain 1 Test accuracy of {domain1_maximum_accuracy} achieved at epoch {epoch}!")

    # save the model weights
    generator_model.save_weights(root_dir + f"checkpoints/test-{epoch}/generator_model/model_weights", overwrite=True, save_format='tf')
    discriminator_model.save_weights(root_dir + f"checkpoints/test-{epoch}/discriminator_model/model_weights", overwrite=True, save_format='tf')
    classifier_model.save_weights(root_dir + f"checkpoints/test-{epoch}/classifier_model/model_weights", overwrite=True, save_format='tf')
    overall_classification_model.save_weights(root_dir + f"checkpoints/test-{epoch}/overall_classification_model/model_weights", overwrite=True, save_format='tf') 

# save the corresponding number of information
with open(root_dir+"saved/num_epochs.txt","w") as f:
    f.write(f"Classification Count: {class0_count}, {class1_count}\n")
    f.write(f"Domain Count: {domain0_count}, {domain1_count}\n")  
    f.write(f"Number of Epochs: {epoch}\n")
    f.write(f"Number of Batches: {batch}\n")
    f.write(f"Maximum Domain 1 Test Accuracy: {domain1_maximum_accuracy}\n")

print("Completed testing!")
print("\nDone! Please continue training if necessary.")
