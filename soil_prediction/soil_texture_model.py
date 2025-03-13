import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define paths to your training and validation image directories.
train_dir = 'C:\\Users\\latha\\OneDrive\\Documents\\Dataset\\test'
val_dir = 'C:\\Users\\latha\\OneDrive\\Documents\\Dataset\\Train'

# Create image data generators for augmentation and rescaling.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Define image size and batch size.
img_size = 224
batch_size = 32

# Create generators from directory.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained MobileNetV2 model without its top classification layers.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze the base model to retain its learned features.
base_model.trainable = False

# Build the custom classifier on top of the base model.
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
# Suppose we have 4 classes: ,black,sandy, red, clay.
predictions = Dense(4, activation='softmax')(x)

# Combine the base model and the custom classifier into one model.
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with an appropriate optimizer and loss function.
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model.
epochs = 10     # Adjust epochs as needed.
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Evaluate the model on the validation set.
loss, accuracy = model.evaluate(val_generator)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))
model.save('C:\\Users\\latha\\PycharmProjects\\hackothon\\my_model.keras')


