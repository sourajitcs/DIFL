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
import models

# initialize the random generator
random.seed()

# model hyperparameters
img_height, img_width = 512,512
secondary_img_height, secondary_img_width = 63,63 
num_of_filters = 512
batch_size = 4
classification_lr = 0.001
generator_lr = discriminator_lr = 0.001
generator_weights = None
frequency = 1
save_frequency = 100
run_number = 0
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
domain1_seed = random.randint(1,100000)
domain2_seed = random.randint(1,100000)

# function to shuffle the domain dataset
def refresh_domain_dataset():
    print("\nRefreshing the domain dataset!")
    dataset = combined_dataset.shuffle(500, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# function to shuffle the classification dataset
def refresh_classification_dataset():
    print("\nRefreshing the classification dataset!")
    dataset = domain1_train_dataset.shuffle(500, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# function to append domain labels to a dataset
def append_domain_label(dataset, label):
    dataset = dataset.unbatch()
    x, y = zip(*dataset)
    nums = len(y)
    x = tf.reshape(x, [nums, img_height, img_width, 3])
    d = [label]*nums
    d = tf.convert_to_tensor(d, dtype=tf.float32)
    x = tf.data.Dataset.from_tensor_slices((x))
    d = tf.data.Dataset.from_tensor_slices(d)
    return tf.data.Dataset.zip((x, d))

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

    '''
    # gradient state updates
    classification_layer1_weights_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[1].layers[2].layers[2].trainable_weights[0])))
    classification_layer1_bias_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[1].layers[2].layers[2].trainable_weights[1])))
    classification_final_layer_weights_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[2].layers[1].layers[-2].trainable_weights[0])))
    classification_final_layer_bias_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[2].layers[1].layers[-2].trainable_weights[1])))
    '''
    del tape

    # update classification accuracy and loss
    classification_accuracy.update_state(classification_y_batch, logits_classification)
    classification_loss.update_state(classification_batch_loss)

    # update the generator and classifier models
    classification_optimizer.apply_gradients(zip(classification_gradients, overall_classification_model.trainable_weights))

    # define the generator labels used for calculating the generator loss
    generator_labels = tf.fill([domain_l,1],0.5)
    #generator_labels = 1-ybatchdomain
    #print(generator_labels)

    # GAN training step
    with tf.GradientTape(persistent=True) as tape:
        encodings_generator = generator_model(domain_x_batch, training=True)
        logits_discriminator = discriminator_model(encodings_generator, training=True)
        discriminator_batch_loss = binary_loss(domain_y_batch, logits_discriminator)
        generator_batch_loss = binary_loss(generator_labels, logits_discriminator)
    discriminator_gradients = tape.gradient(discriminator_batch_loss, discriminator_model.trainable_weights)
    generator_gradients = tape.gradient(generator_batch_loss, generator_model.trainable_weights)
    '''
    # gradient state updates
    domain_layer1_weights_gradients.update_state(tf.abs(tape.gradient(generator_batch_loss, generator_model.layers[2].layers[2].trainable_weights[0])))
    domain_layer1_bias_gradients.update_state(tf.abs(tape.gradient(generator_batch_loss, generator_model.layers[2].layers[2].trainable_weights[1])))
    domain_final_layer_weights_gradients.update_state(tf.abs(tape.gradient(discriminator_batch_loss, discriminator_model.layers[1].layers[-2].trainable_weights[0])))
    domain_final_layer_bias_gradients.update_state(tf.abs(tape.gradient(discriminator_batch_loss, discriminator_model.layers[1].layers[-2].trainable_weights[1])))
    '''
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

if (len(sys.argv)<4):
    print("Usage:")
    print("    python 0_difl.py <<source_dir>> <<target_dir>> <<num_epochs>>")
    sys.exit(1)

# parse input arguments
domain1_directory = sys.argv[1]
domain2_directory = sys.argv[2]
epochs = int(sys.argv[3])

# read the number of runs file
with open("runs/num_of_runs.txt", 'r') as f:
    r = int(f.readline())+1
with open("runs/num_of_runs.txt", 'w') as f:
    f.write(str(r))
'''
# print confirmation screen
print(f"""
Logs are saved as the {prettify_number(r)} run!
Run Number: {run_number}
Start Date & Time: {start_time}
Source Domain: {sys.argv[1]}, Target Domain: {sys.argv[2]}
Image Parameters: {img_height}, {img_width}
Secondary Image Parameters: {secondary_img_height}, {secondary_img_width}
Number of Filters: {num_of_filters}
Batch Size: {batch_size}
Classification Learning Rate: {classification_lr}
Generator & Discriminator Learning Rate: {generator_lr}
Frequency of Output to Tensorboard: {frequency}
Frequency of Saving the DIFL Model: {save_frequency}
Type of Generator Weights: {generator_weights}
Initial Number of Epochs: {sys.argv[3]}
Domain 1 Seed: {domain1_seed}
Domain 2 Seed: {domain2_seed}
""")

if input("Continue?[Y/N]") == 'y':
    pass
else:
    exit()
'''
# parse domain names
temp = domain1_directory.split('/')
if temp[-1] == "":
    temp.pop()
domain1_name = temp[-1]
temp = domain2_directory.split('/')
if temp[-1] == "":
    temp.pop()
domain2_name = temp[-1]

# define the root of the log directory and create the folder
root_dir = f"runs/{domain1_name}--{domain2_name}/{prettify_number(r)}_run_" + start_time + "/"
os.makedirs(root_dir)

# save the model parameters and training information
with open(root_dir + "hyperparameters.txt","w") as f:
    f.write(f"Run Number: {run_number}\n")
    f.write(f"Start Date & Time: {start_time}\n")
    f.write(f"Source Domain: {sys.argv[1]}, Target Domain: {sys.argv[2]}\n")
    f.write(f"Image Parameters: {img_height}, {img_width}\n")
    f.write(f"Secondary Image Parameters: {secondary_img_height}, {secondary_img_width}\n")
    f.write(f"Number of Filters: {num_of_filters}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Classification Learning Rate: {classification_lr}\n")
    f.write(f"Generator & Discriminator Learning Rate: {generator_lr}\n")
    f.write(f"Frequency of output to Tensorboard: {frequency}\n")
    f.write(f"Frequency of Saving the DIFL Model: {save_frequency}\n")
    f.write(f"Type of Generator Weights: {generator_weights}\n")
    f.write(f"Domain 1 Seed: {domain1_seed}\n")
    f.write(f"Domain 2 Seed: {domain2_seed}\n")

print(f"\nImporting the necessary datasets..")

# import the first dataset
domain1_train_dataset = keras.preprocessing.image_dataset_from_directory(domain1_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='training', seed=domain1_seed)
domain1_test_dataset = keras.preprocessing.image_dataset_from_directory(domain1_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='validation', seed=domain1_seed)

# import the second dataset
domain2_train_dataset = keras.preprocessing.image_dataset_from_directory(domain2_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='training', seed=domain2_seed)
domain2_test_dataset = keras.preprocessing.image_dataset_from_directory(domain2_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='validation', seed=domain2_seed)

# calculate the length of the classification dataset
length = len(domain1_train_dataset)

print("Datasets imported!")

# adding domain labels
print("Appending domain labels to Domain1 train..")
d1_train = append_domain_label(domain1_train_dataset, 0)
print("Appending domain labels to Domain2 train..")
d2_train = append_domain_label(domain2_train_dataset, 1)

# create and cache the classification dataset
domain1_train_dataset = domain1_train_dataset.unbatch()
domain1_train_dataset = domain1_train_dataset.cache()

# create and cache the domain dataset
combined_dataset = d1_train.concatenate(d2_train)
combined_dataset = combined_dataset.cache()

# cache the testing datasets
domain1_test_dataset = domain1_test_dataset.cache()
domain2_train_dataset = domain2_train_dataset.cache()
domain2_test_dataset = domain2_test_dataset.cache()

print("Done pre-processing datasets!")

'''
Create the DIFL network models
'''
generator_model = models.create_generator_model(img_height, img_width, generator_weights, num_of_filters)
discriminator_model = models.create_discriminator_model(secondary_img_height, secondary_img_width, num_of_filters)
classifier_model = models.create_classifier_model(secondary_img_height, secondary_img_width, num_of_filters)
overall_classification_model = models.create_overall_classification_model(img_height, img_width, generator_model, classifier_model)

'''
Specify the other parameters for the networks
'''
batch = 0
epoch = 0

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

'''
# instantiate the gradient metrics
classification_layer1_weights_gradients = keras.metrics.MeanTensor()
classification_layer1_bias_gradients = keras.metrics.MeanTensor()
classification_final_layer_weights_gradients = keras.metrics.MeanTensor()
classification_final_layer_bias_gradients = keras.metrics.MeanTensor()
domain_layer1_weights_gradients = keras.metrics.MeanTensor()
domain_layer1_bias_gradients = keras.metrics.MeanTensor()
domain_final_layer_weights_gradients = keras.metrics.MeanTensor()
domain_final_layer_bias_gradients = keras.metrics.MeanTensor()
'''
# define iterators for getting the batches
classification_iterator = iter(refresh_classification_dataset())
domain_iterator = iter(refresh_domain_dataset())

# variable to keep track of maximum accuracy
domain2_maximum_accuracy = 0
overall_maximum_accuracy = 0

'''
Tensorboard initialization
'''
epoch_writer = tf.summary.create_file_writer(root_dir + f"epoch-logs/{run_number}/")
batch_writer = tf.summary.create_file_writer(root_dir + f"batch-logs/{run_number}/")
accuracy_writer = tf.summary.create_file_writer(root_dir + f"accuracy-logs/{run_number}/")

'''
Start training the model as well as its evaluation
'''

# epochs of classification training
while epoch < epochs:
    print(f"Starting epoch {epoch+1}/{epochs} of Domain Discrimination training!")

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
            classification_iterator = iter(refresh_classification_dataset())
            xbatchclass, ybatchclass = classification_iterator.get_next()
            # to view first image of batch to ensure proper shuffling
            #keras.preprocessing.image.save_img(f"class_{epoch}.png", xbatchclass[0], data_format = "channels_last", file_format = "png")

        # get the batches for the domain (GAN) training step
        try:
            xbatchdomain, ybatchdomain = domain_iterator.get_next()
        except tf.errors.OutOfRangeError:
            domain_iterator = iter(refresh_domain_dataset())
            xbatchdomain, ybatchdomain = domain_iterator.get_next()
            # to view first image of batch to ensure proper shuffling
            #keras.preprocessing.image.save_img(f"domain_{epoch}.png", xbatchdomain[0], data_format = "channels_last", file_format = "png")

        # calculate the batch size of the domain set
        domain_length = tf.constant(len(ybatchdomain))

        training_step(xbatchclass, ybatchclass, xbatchdomain, ybatchdomain, domain_length)
        batch += 1

    #print(f"The classification accuracy is {float(classification_accuracy.result())}.")
    #print(f"The domain accuracy is {float(domain_accuracy.result())}.")

    # write to tensorboard
    with epoch_writer.as_default():

        # write the scalar metrics
        tf.summary.scalar("Classification Accuracy", classification_accuracy.result(), step=epoch)
        tf.summary.scalar("Classification Loss", classification_loss.result(), step=epoch)
        tf.summary.scalar("Domain Accuracy", domain_accuracy.result(), step=epoch)
        tf.summary.scalar("Generator Loss", generator_loss.result(), step=epoch)
        tf.summary.scalar("Discriminator Loss", discriminator_loss.result(), step=epoch)
        '''
        # write the gradient state histogram for classification
        tf.summary.histogram("Classification Layer 1 Weight Gradients", classification_layer1_weights_gradients.result(), step=epoch)
        tf.summary.histogram("Classification Layer 1 Bias Gradients", classification_layer1_bias_gradients.result(), step=epoch)
        tf.summary.histogram("Classification Final Layer Weight Gradients", classification_final_layer_weights_gradients.result(), step=epoch)
        tf.summary.histogram("Classification Final Layer Bias Gradients", classification_final_layer_bias_gradients.result(), step=epoch)

        # write the gradient state histogram for domain
        tf.summary.histogram("Generator Layer 1 Weight Gradients", domain_layer1_weights_gradients.result(), step=epoch)
        tf.summary.histogram("Generator Layer 1 Bias Gradients", domain_layer1_bias_gradients.result(), step=epoch)
        tf.summary.histogram("Discriminator Final Layer Weight Gradients", domain_final_layer_weights_gradients.result(), step=epoch)
        tf.summary.histogram("Discriminator Final Layer Bias Gradients", domain_final_layer_bias_gradients.result(), step=epoch)

        # write the state of the networks
        tf.summary.histogram("Generator Layer 1 Weights", generator_model.layers[2].layers[2].trainable_weights[0], step=epoch)
        tf.summary.histogram("Generator Layer 1 Bias", generator_model.layers[2].layers[2].trainable_weights[1], step=epoch)
        tf.summary.histogram("Classifier Final Layer Weights", classification_model.layers[2].layers[1].layers[-2].trainable_weights[0], step=epoch)
        tf.summary.histogram("Classifier Final Layer Bias", classification_model.layers[2].layers[1].layers[-2].trainable_weights[1], step=epoch)
        tf.summary.histogram("Discriminator Final Layer Weights", discriminator_model.layers[1].layers[-2].trainable_weights[0], step=epoch)
        tf.summary.histogram("Discriminator Final Layer Bias", discriminator_model.layers[1].layers[-2].trainable_weights[1], step=epoch)
        '''
    epoch += 1

    # reset the scalar metrics
    classification_accuracy.reset_states()
    classification_loss.reset_states()
    domain_accuracy.reset_states()
    generator_loss.reset_states()
    discriminator_loss.reset_states()
    '''
    # reset the classifcation gradient metrics
    classification_layer1_weights_gradients.reset_states()
    classification_layer1_bias_gradients.reset_states()
    classification_final_layer_weights_gradients.reset_states()
    classification_final_layer_bias_gradients.reset_states()

    # reset the domain gradient metrics
    domain_layer1_weights_gradients.reset_states()
    domain_layer1_bias_gradients.reset_states()
    domain_final_layer_weights_gradients.reset_states()
    domain_final_layer_bias_gradients.reset_states()
    '''
    # reset the classification metric
    classification_accuracy.reset_states()

    # conduct testing at regular intervals
    if epoch%(frequency)==0:

        # test the model on 1st domain train
        for xbatch, ybatch in domain1_train_dataset.batch(batch_size):
            testing_step(xbatch, ybatch)
        domain1_train_accuracy = float(classification_accuracy.result())
        classification_accuracy.reset_states()

        # test the model on 1st domain test
        for xbatch, ybatch in domain1_test_dataset:
            testing_step(xbatch, ybatch)
        domain1_test_accuracy = float(classification_accuracy.result())
        classification_accuracy.reset_states()

        # test the model on 2nd domain train
        for xbatch, ybatch in domain2_train_dataset:
            testing_step(xbatch, ybatch)
        domain2_train_accuracy = float(classification_accuracy.result())
        classification_accuracy.reset_states()

        # test the model on 2nd domain test
        for xbatch, ybatch in domain2_test_dataset:
            testing_step(xbatch, ybatch)
        domain2_test_accuracy = float(classification_accuracy.result())
        classification_accuracy.reset_states()

        # print the result
        print(f"The accuracy on the 1st domain training set is: {domain1_train_accuracy}.")
        print(f"The accuracy on the 1st domain testing set is: {domain1_test_accuracy}.")
        print(f"The accuracy on the 2nd domain training set is: {domain2_train_accuracy}.")
        print(f"The accuracy on the 2nd domain testing set is: {domain2_test_accuracy}.")

        # write results to tensorboard
        with accuracy_writer.as_default():
            tf.summary.scalar("1st Domain Train Accuracy", domain1_train_accuracy, step=epoch)
            tf.summary.scalar("1st Domain Test Accuracy", domain1_test_accuracy, step=epoch)
            tf.summary.scalar("2nd Domain Train Accuracy", domain2_train_accuracy, step=epoch)
            tf.summary.scalar("2nd Domain Test Accuracy", domain2_test_accuracy, step=epoch)

        # check if current results are better than the past best results
        if domain2_test_accuracy > domain2_maximum_accuracy:
            domain2_maximum_accuracy = domain2_test_accuracy
            print(f"Maximum Domain 2 Test accuracy of {domain2_maximum_accuracy} achieved at epoch {epoch}!")

            # save the model weights
            generator_model.save_weights(root_dir + f"checkpoints/test-{epoch}/generator_model/model_weights", overwrite=True, save_format='tf')
            discriminator_model.save_weights(root_dir + f"checkpoints/test-{epoch}/discriminator_model/model_weights", overwrite=True, save_format='tf')
            classifier_model.save_weights(root_dir + f"checkpoints/test-{epoch}/classifier_model/model_weights", overwrite=True, save_format='tf')
            overall_classification_model.save_weights(root_dir + f"checkpoints/test-{epoch}/overall_classification_model/model_weights", overwrite=True, save_format='tf')

        overall_domain2_accuracy = 0.8*domain2_train_accuracy + 0.2*domain2_test_accuracy
        if overall_domain2_accuracy > overall_maximum_accuracy:
            overall_maximum_accuracy = overall_domain2_accuracy
            print(f"Maximum overall accuracy of {overall_maximum_accuracy} achieved at epoch {epoch}!")

            # save the model weights
            generator_model.save_weights(root_dir + f"checkpoints/overall-{epoch}/generator_model/model_weights", overwrite=True, save_format='tf')
            discriminator_model.save_weights(root_dir + f"checkpoints/overall-{epoch}/discriminator_model/model_weights", overwrite=True, save_format='tf')
            classifier_model.save_weights(root_dir + f"checkpoints/overall-{epoch}/classifier_model/model_weights", overwrite=True, save_format='tf')
            overall_classification_model.save_weights(root_dir + f"checkpoints/overall-{epoch}/overall_classification_model/model_weights", overwrite=True, save_format='tf')


    # save the whole model for logkeeping
    if epoch%(save_frequency)==0:
        generator_model.save(root_dir + "saved/generator_model", overwrite=True, include_optimizer=True, save_format='tf')
        discriminator_model.save(root_dir + "saved/discriminator_model", overwrite=True, include_optimizer=True, save_format='tf')
        classifier_model.save(root_dir + "saved/classifier_model", overwrite=True, include_optimizer=True, save_format='tf')
        overall_classification_model.save(root_dir + "saved/overall_classification_model", overwrite=True, include_optimizer=True, save_format='tf')

        # save the corresponding number of epochs
        with open(root_dir+"saved/num_epochs.txt","w") as f:
            f.write(f"Number of Epochs: {epoch}\n")
            f.write(f"Number of Batches: {batch}\n")
            f.write(f"Maximum Overall Accuracy: {overall_maximum_accuracy}\n")
            f.write(f"Maximum Domain 2 Test Accuracy: {domain2_maximum_accuracy}\n")

print("\nDone!")
