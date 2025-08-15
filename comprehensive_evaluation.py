#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import os

def calculate_comprehensive_metrics():
    
    model_path = 'stress_detection_cnn_lstm.h5'
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print("Loading trained model...")
    model = keras.models.load_model(model_path)
    
    print("Generating test data...")
    n_samples = 100
    sequence_length = 240
    n_features = 14
    
    test_data = np.random.randn(n_samples, sequence_length, n_features)
    true_labels = []
    
    for i in range(n_samples):
        if i % 2 == 0:
            test_data[i, :, 0] += np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 0.5
            test_data[i, :, 1] += 80 + np.random.normal(0, 10, sequence_length)
            true_labels.append(1)
        else:
            test_data[i, :, 0] += np.sin(np.linspace(0, 2*np.pi, sequence_length)) * 0.2
            test_data[i, :, 1] += 65 + np.random.normal(0, 5, sequence_length)
            true_labels.append(0)
    
    true_labels = np.array(true_labels)
    
    print("Making predictions...")
    predictions = model.predict(test_data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_probabilities = np.max(predictions, axis=1)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*80)
    
    accuracy = accuracy_score(true_labels, predicted_classes)
    balanced_acc = balanced_accuracy_score(true_labels, predicted_classes)
    
    print(f"\nOVERALL ACCURACY METRICS:")
    print(f"Standard Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    
    f1_macro = f1_score(true_labels, predicted_classes, average='macro')
    f1_micro = f1_score(true_labels, predicted_classes, average='micro')
    f1_weighted = f1_score(true_labels, predicted_classes, average='weighted')
    f1_per_class = f1_score(true_labels, predicted_classes, average=None)
    
    print(f"\nF1 SCORE ANALYSIS:")
    print(f"F1 Score (Macro Average):    {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"F1 Score (Micro Average):    {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"F1 Score (Weighted Average): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"F1 Score Non-Stress (Class 0): {f1_per_class[0]:.4f} ({f1_per_class[0]*100:.2f}%)")
    print(f"F1 Score Stress (Class 1):     {f1_per_class[1]:.4f} ({f1_per_class[1]*100:.2f}%)")
    
    precision_macro = precision_score(true_labels, predicted_classes, average='macro')
    precision_micro = precision_score(true_labels, predicted_classes, average='micro')
    precision_weighted = precision_score(true_labels, predicted_classes, average='weighted')
    precision_per_class = precision_score(true_labels, predicted_classes, average=None)
    
    print(f"\nPRECISION ANALYSIS:")
    print(f"Precision (Macro Average):    {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"Precision (Micro Average):    {precision_micro:.4f} ({precision_micro*100:.2f}%)")
    print(f"Precision (Weighted Average): {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    print(f"Precision Non-Stress (Class 0): {precision_per_class[0]:.4f} ({precision_per_class[0]*100:.2f}%)")
    print(f"Precision Stress (Class 1):     {precision_per_class[1]:.4f} ({precision_per_class[1]*100:.2f}%)")
    
    recall_macro = recall_score(true_labels, predicted_classes, average='macro')
    recall_micro = recall_score(true_labels, predicted_classes, average='micro')
    recall_weighted = recall_score(true_labels, predicted_classes, average='weighted')
    recall_per_class = recall_score(true_labels, predicted_classes, average=None)
    
    print(f"\nRECALL ANALYSIS:")
    print(f"Recall (Macro Average):    {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"Recall (Micro Average):    {recall_micro:.4f} ({recall_micro*100:.2f}%)")
    print(f"Recall (Weighted Average): {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    print(f"Recall Non-Stress (Class 0): {recall_per_class[0]:.4f} ({recall_per_class[0]*100:.2f}%)")
    print(f"Recall Stress (Class 1):     {recall_per_class[1]:.4f} ({recall_per_class[1]*100:.2f}%)")
    
    cm = confusion_matrix(true_labels, predicted_classes)
    print(f"\nCONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"Actual    Non-Stress  Stress")
    print(f"Non-Stress    {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"Stress        {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nCONFUSION MATRIX BREAKDOWN:")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nCLINICAL METRICS:")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity (True Negative Rate): {specificity:.4f} ({specificity*100:.2f}%)")
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"Positive Predictive Value (PPV):  {ppv:.4f} ({ppv*100:.2f}%)")
    print(f"Negative Predictive Value (NPV):  {npv:.4f} ({npv*100:.2f}%)")
    
    print(f"\nCLASS DISTRIBUTION:")
    unique, counts = np.unique(true_labels, return_counts=True)
    for class_val, count in zip(unique, counts):
        class_name = "Non-Stress" if class_val == 0 else "Stress"
        print(f"{class_name} (Class {class_val}): {count} samples ({count/len(true_labels)*100:.1f}%)")
    
    print(f"\nPREDICTION CONFIDENCE ANALYSIS:")
    avg_confidence = np.mean(predicted_probabilities)
    std_confidence = np.std(predicted_probabilities)
    min_confidence = np.min(predicted_probabilities)
    max_confidence = np.max(predicted_probabilities)
    
    print(f"Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"Std Dev Confidence: {std_confidence:.4f} ({std_confidence*100:.2f}%)")
    print(f"Min Confidence:     {min_confidence:.4f} ({min_confidence*100:.2f}%)")
    print(f"Max Confidence:     {max_confidence:.4f} ({max_confidence*100:.2f}%)")
    
    stress_predictions = predicted_classes == 1
    non_stress_predictions = predicted_classes == 0
    
    if np.any(stress_predictions):
        stress_confidence = np.mean(predicted_probabilities[stress_predictions])
        print(f"Avg Confidence (Stress predictions):     {stress_confidence:.4f} ({stress_confidence*100:.2f}%)")
    
    if np.any(non_stress_predictions):
        non_stress_confidence = np.mean(predicted_probabilities[non_stress_predictions])
        print(f"Avg Confidence (Non-Stress predictions): {non_stress_confidence:.4f} ({non_stress_confidence*100:.2f}%)")
    
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 60)
    class_names = ['Non-Stress', 'Stress']
    report = classification_report(true_labels, predicted_classes, target_names=class_names, digits=4)
    print(report)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    metrics_summary = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'f1_non_stress': f1_per_class[0],
        'f1_stress': f1_per_class[1],
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    return metrics_summary

if __name__ == "__main__":
    metrics = calculate_comprehensive_metrics()
