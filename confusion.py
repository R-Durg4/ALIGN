import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model('model/fitness_posture_model.keras')

# Define image preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

# Load annotations
df = pd.read_csv('test/annotations.csv')
image_dir = 'test/images'
class_names = ['squat', 'benchpress', 'deadlift']
label_to_index = {name: idx for idx, name in enumerate(class_names)}

# Prepare data
y_true = []
y_pred = []

for _, row in df.iterrows():
    filename = row['filename']
    true_label = row['label']
    img_path = os.path.join(image_dir, filename)
    
    img = preprocess_image(img_path)
    if img is None:
        continue

    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred_label = class_names[np.argmax(pred)]

    y_true.append(true_label)
    y_pred.append(pred_label)

# Generate confusion matrix and accuracy
cm = confusion_matrix(y_true, y_pred, labels=class_names)
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

# Display and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_correct.png')
plt.show()

# Output results
print("Classification Report:")
print(report)
print(f"\n✅ Accuracy: {acc:.2%}")
print("\nConfusion Matrix:")
print(cm)   