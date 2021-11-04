import os
import sys
import time
import datetime
import random
import math
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

# initialize the random generator
random.seed()
gpu = 0

# function to shuffle the domain dataset
def shuffle_domain_directory():
    
    print("\nShuffling the domain dataset!")
    
    # reset the domain directory
    try:
        shutil.rmtree(f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/") 
    except:
        pass
    os.makedirs(f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/0/")
    os.makedirs(f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/1/")
    print("Domain directory reset!")

    # helper variables for shuffling the files
    domain0_names = list(range(domain0_count))
    random.shuffle(domain0_names)
    multiplier = math.ceil(domain0_count/domain1_count)
    domain1_names = list(range(domain1_count*multiplier))
    random.shuffle(domain1_names)
    d0 = d1 = 0

    # copy files to folder to create domain dataset
    for i in range(5):
        if i != domain0_seed:
            for f in glob.glob(f"{domain0_directory}/{i}/0/*"):
                shutil.copy2(f, f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/0/{domain0_names[d0]}.png")
                d0 += 1
            for f in glob.glob(f"{domain0_directory}/{i}/1/*"):
                shutil.copy2(f, f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/0/{domain0_names[d0]}.png")
                d0 += 1
    for m in range(multiplier):
        for i in range(5):
            if i != domain1_seed:
                for f in glob.glob(f"{domain1_directory}/{i}/0/*"):
                    shutil.copy2(f, f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/1/{domain1_names[d1]}.png")
                    d1 += 1
                for f in glob.glob(f"{domain1_directory}/{i}/1/*"):
                    shutil.copy2(f, f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/1/{domain1_names[d1]}.png")
                    d1 += 1
       
    print("Done shifting files for domain dataset!")

# function to shuffle the classification dataset
def shuffle_classification_directory():
    
    print("\nShuffling the classification dataset!")

    # reset the classification directory
    try:
        shutil.rmtree(f"../datasets/domains-{gpu}/{domain0_name}/train/")
    except:
        pass
    os.makedirs(f"../datasets/domains-{gpu}/{domain0_name}/train/0/")
    os.makedirs(f"../datasets/domains-{gpu}/{domain0_name}/train/1/") 
    print("Training classification directory reset!")

    # helper variables for shuffling the files
    class0_names = list(range(class0_count))
    random.shuffle(class0_names)
    class1_names = list(range(class1_count))
    random.shuffle(class1_names)
    c0 = c1 = 0

    # copy files to folder to create classification datasets
    for i in range(5):
        if i != domain0_seed:
            for f in glob.glob(f"{domain0_directory}/{i}/0/*"):
                shutil.copy2(f, f"../datasets/domains-{gpu}/{domain0_name}/train/0/{class0_names[c0]}.png")
                c0 += 1
            for f in glob.glob(f"{domain0_directory}/{i}/1/*"):
                shutil.copy2(f, f"../datasets/domains-{gpu}/{domain0_name}/train/1/{class1_names[c1]}.png")
                c1 += 1
    
    print("Done shifting files for classification domain 0 training dataset!")
 
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
Continuing the run in the {root_dir} folder!
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

# accept user input for adjusting training parameters
adjust = "X"
while adjust != "Y" and adjust != "y" and adjust != "N" and adjust != "n":
    adjust = input("Adjust training parameters?[Y/N]")
if adjust == "Y" or adjust == "y":
    classification_lr = float(input("Enter the classification learning rate: "))
    generator_lr = discriminator_lr = float(input("Enter the generator and discriminator learning rate: "))

# print confirmation screen
print(f"""
Continuing the run in the {root_dir} folder!
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

# save the model parameters and training information
with open(root_dir + "hyperparameters.txt","w") as f:
    f.write(f"Start Date & Time: {start_time}\n")
    f.write(f"Source Domain: {domain0_directory}, Target Domain: {domain1_directory}\n")
    f.write(f"Image Parameters: {img_height}, {img_width}\n")
    f.write(f"Secondary Image Parameters: {secondary_img_height}, {secondary_img_width}\n")
    f.write(f"Number of Filters: {num_of_filters}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Classification Learning Rate: {classification_lr}\n")
    f.write(f"Generator & Discriminator Learning Rate: {generator_lr}\n")
    f.write(f"Type of Generator Weights: {generator_weights}\n")
    f.write(f"Domain 0 Seed: {domain0_seed}\n")
    f.write(f"Domain 1 Seed: {domain1_seed}\n")

print(f"\nImporting the necessary datasets..")

# shuffle the classification and domain directories
shuffle_classification_directory()
shuffle_domain_directory()

# create the classification datasets
domain0_train_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain0_name}/train/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
domain0_train_dataset = domain0_train_dataset.cache()
domain0_train_dataset = domain0_train_dataset.prefetch(tf.data.AUTOTUNE)

# calculate the length of the classification dataset
length = len(domain0_train_dataset)

# create the combined dataset
combined_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
combined_dataset = combined_dataset.cache()
combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)

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

# instantiate the optimizer for each network
generator_optimizer = keras.optimizers.SGD(learning_rate=generator_lr, momentum=0.9)
discriminator_optimizer = keras.optimizers.SGD(learning_rate=discriminator_lr, momentum=0.9)
classification_optimizer = keras.optimizers.SGD(learning_rate=classification_lr, momentum=0.9)

# instantiate the loss function
binary_loss = keras.losses.BinaryCrossentropy()

# instantiate the accuracy metrics
domain_accuracy = keras.metrics.BinaryAccuracy()
classification_accuracy = keras.metrics.BinaryAccuracy()

# instantiate the loss metrics
discriminator_loss = keras.metrics.Mean()
generator_loss = keras.metrics.Mean()
classification_loss = keras.metrics.Mean()

# define iterators for getting the batches
classification_iterator = iter(domain0_train_dataset)
domain_iterator = iter(combined_dataset)

'''
Tensorboard initialization
'''
epoch_writer = tf.summary.create_file_writer(root_dir + f"epoch-logs/")
#batch_writer = tf.summary.create_file_writer(root_dir + f"batch-logs/")
accuracy_writer = tf.summary.create_file_writer(root_dir + f"accuracy-logs/")

'''
Start training the model as well as its evaluation
'''
# main training loop
print(f"Starting the epoch!")

# custom training loop for each batch in the training dataset
for i in range(length):
    
    if i == length-1:
        print(f"Training batch {i+1}...  ")
    elif i%3 == 0:
        print(f"Training batch {i+1}...  ", end='\r')
    elif i%3 == 1:
        print(f"Training batch {i+1}.    ", end='\r')
    else:
        print(f"Training batch {i+1}..   ", end='\r')
    
    # get the batches for the classification training step
    try:
        xbatchclass, ybatchclass = classification_iterator.get_next()
    except tf.errors.OutOfRangeError:
        shuffle_classification_directory()
        domain0_train_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain0_name}/train/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
        domain0_train_dataset = domain0_train_dataset.cache()
        domain0_train_dataset = domain0_train_dataset.prefetch(tf.data.AUTOTUNE)
        classification_iterator = iter(domain0_train_dataset)
        xbatchclass, ybatchclass = classification_iterator.get_next()
    
    # to view first image of batch to ensure proper shuffling
    #if i == 0:
    #    keras.preprocessing.image.save_img(f"class_{epoch}.png", xbatchclass[0], data_format = "channels_last", file_format = "png")
    
    # get the batches for the domain (GAN) training step
    try:
        xbatchdomain, ybatchdomain = domain_iterator.get_next()
    except tf.errors.OutOfRangeError:
        shuffle_domain_directory()
        combined_dataset = keras.preprocessing.image_dataset_from_directory(f"../datasets/domains-{gpu}/{domain0_name}--{domain1_name}/", labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width))
        combined_dataset = combined_dataset.cache()
        combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)
        domain_iterator = iter(combined_dataset)
        xbatchdomain, ybatchdomain = domain_iterator.get_next()
        
    # to view first image of batch to ensure proper shuffling
    #if i == 0:
    #    keras.preprocessing.image.save_img(f"domain_{epoch}.png", xbatchdomain[0], data_format = "channels_last", file_format = "png")

    # calculate the batch size of the domain set
    domain_length = tf.constant(len(ybatchdomain))
    
    training_step(xbatchclass, ybatchclass, xbatchdomain, ybatchdomain, domain_length)
    batch += 1

# write to tensorboard
with epoch_writer.as_default():
    
    # write the scalar metrics
    tf.summary.scalar("Classification Accuracy", classification_accuracy.result(), step=epoch)
    tf.summary.scalar("Classification Loss", classification_loss.result(), step=epoch)
    tf.summary.scalar("Domain Accuracy", domain_accuracy.result(), step=epoch)
    tf.summary.scalar("Generator Loss", generator_loss.result(), step=epoch)
    tf.summary.scalar("Discriminator Loss", discriminator_loss.result(), step=epoch)
   
epoch += 1

# reset the scalar metrics
classification_accuracy.reset_states()
classification_loss.reset_states()
domain_accuracy.reset_states()
generator_loss.reset_states()
discriminator_loss.reset_states()
''' 
# conduct testing at regular intervals
if epoch%(frequency)==0: 

    # test the model on 1st domain test
    for xbatch, ybatch in domain1_test_dataset:
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
'''
# save the whole model for logkeeping
generator_model.save(root_dir + "saved/generator_model", overwrite=True, include_optimizer=True, save_format='tf')
discriminator_model.save(root_dir + "saved/discriminator_model", overwrite=True, include_optimizer=True, save_format='tf')
classifier_model.save(root_dir + "saved/classifier_model", overwrite=True, include_optimizer=True, save_format='tf')
overall_classification_model.save(root_dir + "saved/overall_classification_model", overwrite=True, include_optimizer=True, save_format='tf')

# save the corresponding number of epochs
with open(root_dir+"saved/num_epochs.txt","w") as f:
    f.write(f"Classification Count: {class0_count}, {class1_count}\n")
    f.write(f"Domain Count: {domain0_count}, {domain1_count}\n")  
    f.write(f"Number of Epochs: {epoch}\n")
    f.write(f"Number of Batches: {batch}\n")
    f.write(f"Maximum Domain 1 Test Accuracy: {domain1_maximum_accuracy}\n")

print("Completed one epoch!")

print("\nDone! Please do testing if necessary.")
