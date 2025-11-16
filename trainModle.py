import tensorflow as tf
from tensorflow.keras import layers, models
import os


data_dir = r"C:\Users\Asus\Desktop\Data"

batch_size = 32 
img_size = (300, 300)  

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,  
    subset="training",  
    seed=123  
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    subset="validation",  
    seed=123
)


class_names = os.listdir(data_dir)
print("Class Names:", class_names)


normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))


data_augmentation = tf.keras.Sequential([
    layers.RandomZoom(0.2)
])


model = models.Sequential([
    layers.Input(shape=(300, 300, 3)),  
    data_augmentation, 
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  
])


model.compile(
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']
)


model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)


model.save("fit.h5")

print("Model training complete and saved as 'fit.h5'")
