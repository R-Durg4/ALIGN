import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import pandas as pd
from PIL import Image
import cv2

def generate_confusion_matrix_from_annotations(model_path, annotations_csv_path, images_dir, batch_size=32):
    """
    Generate and visualize confusion matrix for the fitness posture model using an annotations CSV file
    
    Args:
        model_path: Path to the saved model
        annotations_csv_path: Path to the annotations CSV file
        images_dir: Directory containing the image files
        batch_size: Batch size for prediction
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load annotations
    print(f"Loading annotations from {annotations_csv_path}...")
    annotations_df = pd.read_csv(annotations_csv_path)
    
    # Identify columns in the CSV
    print("Analyzing annotations file structure...")
    columns = annotations_df.columns.tolist()
    print(f"Found columns: {columns}")
    
    # Determine appropriate column names (this will need adjustment based on your actual CSV structure)
    if 'filename' in columns:
        filename_col = 'filename'
    elif 'image_path' in columns:
        filename_col = 'image_path'
    elif 'image' in columns:
        filename_col = 'image'
    else:
        # Try to find a column that might contain filenames
        for col in columns:
            if annotations_df[col].dtype == 'object' and '.jpg' in str(annotations_df[col].iloc[0]) or '.png' in str(annotations_df[col].iloc[0]):
                filename_col = col
                break
        else:
            raise ValueError("Could not find a column containing image filenames in the annotations CSV")
    
    if 'label' in columns:
        label_col = 'label'
    elif 'class' in columns:
        label_col = 'class'
    elif 'category' in columns:
        label_col = 'category'
    elif 'exercise_type' in columns:
        label_col = 'exercise_type'
    else:
        # Try to find a column that might contain class labels
        for col in columns:
            if col != filename_col and annotations_df[col].dtype == 'object':
                # Check if values look like class names
                unique_vals = annotations_df[col].unique()
                if len(unique_vals) < 20:  # Assume fewer than 20 classes
                    label_col = col
                    break
        else:
            raise ValueError("Could not find a column containing class labels in the annotations CSV")
    
    print(f"Using '{filename_col}' as the image filename column")
    print(f"Using '{label_col}' as the class label column")
    
    # Get unique class labels
    class_names = sorted(annotations_df[label_col].unique())
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create class mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Prepare data for prediction
    X = []  # Images
    y_true = []  # True labels
    
    print(f"Processing {len(annotations_df)} images...")
    for idx, row in annotations_df.iterrows():
        # Get image path
        image_path = row[filename_col]
        
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join(images_dir, image_path)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get true label
        label = row[label_col]
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            
            X.append(img)
            y_true.append(class_to_idx[label])
            
            # Print progress
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(annotations_df)} images")
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    if not X:
        raise ValueError("No valid images found for processing")
    
    # Convert to numpy arrays
    X = np.array(X)
    y_true = np.array(y_true)
    
    print(f"Making predictions on {len(X)} images...")
    # Predict in batches to avoid memory issues
    predictions = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_preds = model.predict(batch_X)
        predictions.append(batch_preds)
        
        # Print progress
        if (i + batch_size) % 200 == 0:
            print(f"Predicted {i + batch_size}/{len(X)} images")
    
    # Combine batch predictions
    predictions = np.vstack(predictions)
    y_pred = np.argmax(predictions, axis=1)
    
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    
    # Use a normalized confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix for Fitness Posture Model', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('fitness_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Print a classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate overall accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Analyze misclassifications
    analyze_misclassifications(y_true, y_pred, class_names)
    
    # Also return the raw data for further analysis if needed
    return {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names,
        'predictions': predictions
    }

def analyze_misclassifications(y_true, y_pred, class_names):
    """
    Analyze the most common misclassifications
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        class_names: List of class names
    """
    misclassified = y_true != y_pred
    misclassified_indices = np.where(misclassified)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    
    # Count the most common types of misclassifications
    misclass_types = {}
    for idx in misclassified_indices:
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        misclass_pair = (true_class, pred_class)
        
        if misclass_pair in misclass_types:
            misclass_types[misclass_pair] += 1
        else:
            misclass_types[misclass_pair] = 1
    
    # Sort by frequency
    sorted_misclass = sorted(misclass_types.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost Common Misclassifications:")
    print("-------------------------------")
    for (true_class, pred_class), count in sorted_misclass[:10]:  # Show top 10
        print(f"True: {true_class}, Predicted: {pred_class} - {count} instances")
    
    # Calculate the percentage of misclassifications
    total_samples = len(y_true)
    total_misclassified = len(misclassified_indices)
    misclass_percent = (total_misclassified / total_samples) * 100
    print(f"\nTotal samples: {total_samples}")
    print(f"Total misclassified: {total_misclassified} ({misclass_percent:.2f}%)")
    
    # Print exercise-specific analysis
    print("\nPer-Exercise Analysis:")
    print("---------------------")
    for i, class_name in enumerate(class_names):
        # Calculate precision and recall for this class
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) == 0:
            print(f"{class_name}: No samples in test set")
            continue
            
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  - Samples: {len(class_indices)}")
        print(f"  - Correctly classified: {true_positives} ({(true_positives/len(class_indices))*100:.2f}%)")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        # Most confused with
        if false_negatives > 0:
            incorrect_preds = y_pred[np.where((y_true == i) & (y_pred != i))[0]]
            confused_with = {}
            for pred in incorrect_preds:
                pred_name = class_names[pred]
                confused_with[pred_name] = confused_with.get(pred_name, 0) + 1
            
            top_confused = sorted(confused_with.items(), key=lambda x: x[1], reverse=True)[:3]
            print("  - Most confused with:", ", ".join([f"{name} ({count})" for name, count in top_confused]))
        print()

def visualize_confidence_distribution(predictions, y_true, class_names):
    """
    Visualize the distribution of prediction confidences
    
    Args:
        predictions: Raw prediction probabilities
        y_true: Array of true labels
        class_names: List of class names
    """
    # Get the highest confidence for each prediction
    max_confidences = np.max(predictions, axis=1)
    
    # Separate confidences for correct and incorrect predictions
    y_pred = np.argmax(predictions, axis=1)
    correct_mask = y_pred == y_true
    
    correct_confidences = max_confidences[correct_mask]
    incorrect_confidences = max_confidences[~correct_mask]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(correct_confidences, bins=20, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(incorrect_confidences, bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Number of Predictions', fontsize=12)
    plt.title('Distribution of Model Confidence Scores', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("Confidence distribution visualization saved as 'confidence_distribution.png'")
    
    # Calculate and print statistics
    avg_correct_conf = np.mean(correct_confidences) if len(correct_confidences) > 0 else 0
    avg_incorrect_conf = np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0
    
    print("\nConfidence Score Analysis:")
    print(f"Average confidence for correct predictions: {avg_correct_conf:.4f}")
    print(f"Average confidence for incorrect predictions: {avg_incorrect_conf:.4f}")
    
    # Per-class confidence analysis
    print("\nPer-Class Confidence Analysis:")
    for i, class_name in enumerate(class_names):
        # Get indices where true label is this class
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) == 0:
            continue
            
        class_predictions = predictions[class_indices]
        class_max_confidences = np.max(class_predictions, axis=1)
        class_pred_labels = np.argmax(class_predictions, axis=1)
        
        # Correct predictions for this class
        correct_indices = class_pred_labels == i
        
        avg_confidence = np.mean(class_max_confidences)
        avg_correct_conf = np.mean(class_max_confidences[correct_indices]) if np.any(correct_indices) else 0
        
        print(f"{class_name}:")
        print(f"  - Average confidence: {avg_confidence:.4f}")
        print(f"  - Average confidence when correct: {avg_correct_conf:.4f}")

# Example usage
if __name__ == "__main__":
    model_path = "fitness_posture_model.keras"
    annotations_csv_path = "train\_annotations.csv"
    images_dir = "train\images"  # Directory containing the image files
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found! Please provide the correct path.")
        exit(1)
    
    # Check if annotations CSV exists
    if not os.path.exists(annotations_csv_path):
        print(f"Annotations CSV file {annotations_csv_path} not found! Please provide the correct path.")
        exit(1)
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} not found! Please provide the correct path.")
        exit(1)
    
    # Generate the confusion matrix
    print("Generating confusion matrix...")
    results = generate_confusion_matrix_from_annotations(model_path, annotations_csv_path, images_dir)
    
    # Visualize confidence distribution
    print("\nVisualizing confidence distribution...")
    visualize_confidence_distribution(results['predictions'], results['y_true'], results['class_names'])
    
    print("\nAnalysis complete! Confusion matrix saved as 'fitness_model_confusion_matrix.png'")



