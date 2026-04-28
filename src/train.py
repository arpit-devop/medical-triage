import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ensure src/ is on the path so relative imports work from any cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data.preprocessing import DataPreprocessor
from models.lstm_model import build_lstm_model
from models.bert_model import BioBERTClassifier
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

class MedicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_real_data(data_path):
    save_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Using dummy data.")
        return load_dummy_data()
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Kaggle medicaltranscriptions dataset has 'transcription' and 'medical_specialty'
    if 'transcription' in df.columns and 'medical_specialty' in df.columns:
        df = df.dropna(subset=['transcription', 'medical_specialty'])
        texts = df['transcription'].tolist()
        labels_text = df['medical_specialty'].tolist()
    else:
        # Fallback if different dataset format
        print("Expected columns not found. Please ensure 'transcription' and 'medical_specialty' columns exist.")
        return load_dummy_data()
        
    le = LabelEncoder()
    labels = le.fit_transform(labels_text)
    
    # Save label encoder classes for later inference
    np.save(os.path.join(PROJECT_ROOT, 'saved_models', 'classes.npy'), le.classes_)
    
    return texts, labels, len(le.classes_)

def load_dummy_data():
    texts = [
        "Severe chest pain and shortness of breath",
        "Chronic migraines with aura",
        "Fractured femur after fall",
        "Severe heartburn and stomach pain",
        "Asthma attack and wheezing",
        "Routine checkup for blood pressure"
    ]
    labels = [0, 1, 2, 3, 4, 5]
    save_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'classes.npy'), np.array(['Cardiology', 'Neurology', 'Orthopedics', 'Gastroenterology', 'Pulmonology', 'General']))
    return texts, labels, 6

def train_lstm(data_path):
    print("Preparing data for LSTM...")
    processor = DataPreprocessor()
    texts, labels, num_classes = load_real_data(data_path)
    
    X = processor.prepare_lstm_data(texts)
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_lstm_model(num_classes=num_classes)
    
    print("Training LSTM model...")
    save_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'lstm_model.h5')
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)
    
    # Save the tokenizer
    import pickle
    tokenizer_path = os.path.join(save_dir, 'lstm_tokenizer.pkl')
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(processor.lstm_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Training complete. Model saved to {model_path}")

def train_bert(data_path):
    print("Preparing data for BioBERT...")
    processor = DataPreprocessor()
    texts, labels, num_classes = load_real_data(data_path)
    
    encodings = processor.prepare_bert_data(texts)
    
    # Train test split
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
    
    train_encodings = {key: val[train_idx] for key, val in encodings.items()}
    val_encodings = {key: val[val_idx] for key, val in encodings.items()}
    
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    
    train_dataset = MedicalDataset(train_encodings, train_labels)
    val_dataset = MedicalDataset(val_encodings, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = BioBERTClassifier(num_classes=num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()
    
    epochs = 3
    print("Training BioBERT model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels_tensor)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Average Training Loss: {total_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_tensor = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == labels_tensor).sum().item()
                total += labels_tensor.size(0)
                
        print(f"Validation Accuracy: {correct / total:.4f}")
        
    print("Training complete.")
    bert_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', 'biobert_finetuned')
    os.makedirs(bert_save_dir, exist_ok=True)
    model.bert.save_pretrained(bert_save_dir)
    processor.bert_tokenizer.save_pretrained(bert_save_dir)
    print(f"Model saved to {bert_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "bert"], default="lstm", help="Which model to train")
    default_data = os.path.join(PROJECT_ROOT, 'data', 'mtsamples.csv')
    parser.add_argument("--data", type=str, default=default_data, help="Path to CSV data")
    args = parser.parse_args()
    
    if args.model == "lstm":
        train_lstm(args.data)
    else:
        train_bert(args.data)
