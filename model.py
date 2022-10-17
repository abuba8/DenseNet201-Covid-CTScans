from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras import Model


IMG_SIZE = 224

def feature_extractor():
    #initializing densenet model
    base_model = DenseNet201(weights='imagenet', include_top=True, input_shape=(224,224,3), pooling=None)
    # base_model.summary()

    #freezing layers
    for layer in base_model.layers: 
        layer.trainable = False
        print('Layer ' + layer.name + ' frozen.')
        
        
    #adding last layer with 1 neuron
    last = base_model.layers[-2].output
    x = Dense(1, activation='sigmoid', name='output_final_layer')(last)
    model = Model(base_model.input, x)
    return model

def fine_tune():
    base_model = DenseNet201(weights='imagenet', include_top=True, input_shape=(224,224,3), pooling=None)
    
    #freezing layers
    for layer in base_model.layers: 
        layer.trainable = False
        print('Layer ' + layer.name + ' frozen.')


    #adding three dense layers
    last = base_model.layers[-2].output
    x = Dense(128, activation='relu', name='fc1')(last)
    x = Dense(64, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='output_final_layer')(x)
    model = Model(base_model.input, x)
    return model