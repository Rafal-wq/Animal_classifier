import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'animal_model.h5')
TEST_DIR = os.path.join(BASE_DIR, 'dataset_test')
CLASS_FILE = os.path.join(BASE_DIR, 'class_names.txt')

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# load class names
with open(CLASS_FILE) as f:
    class_names = [line.strip() for line in f.readlines()]

# dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# true labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# predictions
preds = model.predict(test_ds)
y_pred = np.argmax(preds, axis=1)

# accuracy
accuracy = np.mean(y_pred == y_true)
print("\nTEST ACCURACY:", round(accuracy, 4))

# classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
