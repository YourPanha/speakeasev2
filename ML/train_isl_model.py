import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Training ISL (Indian Sign Language) Model...")

# Directory setup
base_dir = "ISL_dataset"
output_model_path = "isl_model.h5"

# Check if dataset exists
if not os.path.exists(base_dir):
    print(f"Error: Dataset directory {base_dir} not found!")
    print("Please run data_collection_isl.py to collect ISL dataset first.")
    exit(1)

# Check if enough data is available
min_images_per_class = 30
enough_data = True

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    char_dir = os.path.join(base_dir, char)
    if not os.path.exists(char_dir) or len(os.listdir(char_dir)) < min_images_per_class:
        print(f"Warning: Class {char} has insufficient data (< {min_images_per_class} images).")
        enough_data = False

if not enough_data:
    print("\nNot enough training data for some classes.")
    response = input("Do you want to continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting. Please collect more data using data_collection_isl.py")
        exit(1)

# Data preparation
print("\nPreparing dataset...")

# Data augmentation to increase variation in the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    validation_split=0.2  # Set aside 20% of data for validation
)

# Use same preprocessing for validation but no augmentation
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Parameters
img_width, img_height = 400, 400
batch_size = 32
epochs = 50
num_classes = 26  # A-Z

# Load the data
training_set = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

validation_set = validation_datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

print(f"\nTraining set: {training_set.samples} images")
print(f"Validation set: {validation_set.samples} images")
print(f"Class mapping: {training_set.class_indices}")

# Build CNN model
print("\nBuilding model...")
model = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Third Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Fourth Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and fully connected layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout for regularization
    Dense(num_classes, activation='softmax')  # Output layer for 26 classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Callbacks for model training
checkpoint = ModelCheckpoint(
    output_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train the model
print("\nTraining model...")
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // batch_size,
    epochs=epochs,
    validation_data=validation_set,
    validation_steps=validation_set.samples // batch_size,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the final model
model.save(output_model_path.replace('.h5', '_final.h5'))
print(f"\nModel saved as {output_model_path} (best) and {output_model_path.replace('.h5', '_final.h5')} (final)")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('isl_training_history.png')
plt.show()

print("\nISL model training complete!")
print("You can now use this model in your sign language app by setting the isl_model_path in final_pred.py") 