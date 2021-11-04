import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def create_generator_model(img_height, img_width, generator_weights, num_of_filters):
    
    # define the base resnet
    res = keras.applications.ResNet50(include_top=False,input_shape=(img_height,img_width,3),weights=generator_weights)
    
    # define the entire network architecture
    inputs = keras.Input(shape=(img_height,img_width,3))
    x = Rescaling(scale=1.0/255)(inputs)
    x = res(x)
    x = layers.Conv2DTranspose(1024, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Conv2DTranspose(num_of_filters, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Conv2D(num_of_filters, (3,3), activation= layers.LeakyReLU(alpha=0.2))(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2DTranspose(num_of_filters, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Conv2D(num_of_filters, (3,3), activation= layers.LeakyReLU(alpha=0.2))(x)
    x = layers.MaxPooling2D()(x)
    outputs = layers.Conv2DTranspose(num_of_filters, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
    
    # define the final model
    generator_model = keras.Model(inputs=inputs, outputs=outputs, name="DIFL_Generator_Model")
    
    # display the difl model summary
    print(generator_model.summary())

    return generator_model

def create_discriminator_model(secondary_img_height, secondary_img_width, num_of_filters):

    # define the base vgg model
    vgg = keras.applications.VGG19(include_top=False,input_shape=(secondary_img_height,secondary_img_width,num_of_filters),weights=None)

    # define the entire network architecture
    inputs = keras.Input(shape=(secondary_img_height,secondary_img_width,num_of_filters))
    x = vgg(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(128, activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.2))(x)
    outputs = layers.Dense(1, activation = 'sigmoid')(x)

    # define the final model
    discriminator_model = keras.Model(inputs=inputs, outputs=outputs, name="Discriminator_Model")

    # display the model summary
    print(discriminator_model.summary())

    return discriminator_model


def create_classifier_model(secondary_img_height, secondary_img_width, num_of_filters):

    # define the base vgg network
    vgg = keras.applications.VGG19(include_top=False, input_shape=(secondary_img_height,secondary_img_width,num_of_filters), weights=None)

    # define the entire network architecture
    inputs = keras.Input(shape=(secondary_img_height,secondary_img_width,num_of_filters))
    x = vgg(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(128, activation = layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.2))(x)
    outputs = layers.Dense(1, activation = 'sigmoid')(x)

    # define the final model
    classifier_model = keras.Model(inputs=inputs, outputs=outputs, name="Classifier_Model")

    # display the model summary
    print(classifier_model.summary())

    return classifier_model

def create_overall_classification_model(img_height, img_width, generator_model, classifier_model):

    # define the entire network architecture
    inputs = keras.Input(shape=(img_height,img_width,3))
    x = generator_model(inputs)
    outputs = classifier_model(x)

    # define the final model
    overall_classification_model = keras.Model(inputs=inputs, outputs=outputs, name="Overall_Classification_Model")

    # display the model summary
    print(overall_classification_model.summary())

    return overall_classification_model
