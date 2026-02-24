# train.py
import tensorflow as tf
from model import (build_custom_cnn, build_resnet_model, 
                   build_efficientnet_model, compile_model, get_callbacks)
from utils import get_data_generators, plot_training_history, evaluate_and_visualize
import os

# --- GPU Check ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU.")

# --- 1. Load Data ---
print("Loading Data...")
train_gen, val_gen, test_gen = get_data_generators()

# --- 2. Select Model to Train ---
# Options: 'cnn', 'resnet', 'efficientnet'
TRAIN_MODEL = 'resnet' 

print(f"\n--- Starting Training for: {TRAIN_MODEL.upper()} ---")

if TRAIN_MODEL == 'cnn':
    model = build_custom_cnn()
    compile_model(model, learning_rate=0.001)
    epochs = 30
    model_name = "CustomCNN"

elif TRAIN_MODEL == 'resnet':
    model, base_model = build_resnet_model()
    compile_model(model, learning_rate=0.001)
    
    # --- Phase 1: Train only the head ---
    print("Phase 1: Training Top Layers...")
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=get_callbacks('resnet_p1')
    )

    # --- Phase 2: Fine-tuning ---
    print("\nPhase 2: Fine-tuning ResNet50 (Unfreezing last 10 layers)...")
    base_model.trainable = True
    # Freeze all except last 10 layers
    for layer in base_model.layers[:-10