import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

# Set your data directory
data_dir = '/Users/macpro/Documents/PLANT DETECTION/KAGGLE/training'


# Set image size
img_size = (224, 224)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

# Load DenseNet with pre-trained weights (without top layers)
base_model = DenseNet121(weights='/Users/macpro/Documents/PLANT DETECTION/KAGGLE/DENSENET/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))


# Add your custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(11, activation='softmax')(x)  # Assuming 11 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the trained model
model.save('densenet_training_model.h5')