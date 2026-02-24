# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, 
                                     Dropout, BatchNormalization, Input, 
                                     GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, EfficientNetB0
import os

def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Model 1: Custom CNN from scratch.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Fully Connected
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary Output
    ])
    return model

def build_resnet_model(input_shape=(224, 224, 3)):
    """
    Model 2: Transfer Learning using ResNet50.
    """
    # Load base model (ImageNet weights)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base initially
    base_model.trainable = False

    # Build classification head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model, base_model

def build_efficientnet_model(input_shape=(224, 224, 3)):
    """
    Model 3: Transfer Learning using EfficientNetB0.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, base_model)
    return model, base_model

def compile_model(model, learning_rate=0.001):
    """
    Compiles the model with Adam optimizer and Binary Crossentropy.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def get_callbacks(model_name):
    """
    Returns standard training callbacks.
    """
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f'{model_name}_best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]