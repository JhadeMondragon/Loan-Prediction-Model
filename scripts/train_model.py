import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/Loan_default.csv')
MODEL_DIR = os.path.join(BASE_DIR, '../backend/models')
RESULTS_DIR = os.path.join(BASE_DIR, '../results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        data = pd.read_csv(filepath)
        if 'LoanID' in data.columns:
            data = data.drop(columns=['LoanID'])
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        exit(1)

def get_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    print(f"Numeric columns: {list(numeric_cols)}")
    print(f"Categorical columns: {list(categorical_cols)}")

    # Numeric Pipeline: Impute Mean -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline: Impute Constant -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor, list(numeric_cols), list(categorical_cols)

def train_xgboost(X_train, y_train, preprocessor):
    print("Training XGBoost...")
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def train_logistic(X_train, y_train, preprocessor):
    print("Training Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

# --- Neural Network Section ---

class MultiLayerNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class SklearnNNWrapper:
    """
    Wrapper to make PyTorch model compatible with sklearn Pipeline.
    """
    def __init__(self, input_dim, output_dim, device, epochs=20, lr=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.model = None
        
    def fit(self, X, y):
        # Convert to tensors
        # X is typically a numpy array after preprocessing
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long).to(self.device)
        
        self.model = MultiLayerNet(self.input_dim, self.output_dim).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

def train_neural_network(X_train, y_train, preprocessor, device):
    print("Training Neural Network...")
    
    # We need to fit the preprocessor first to know the input dimension
    X_train_processed = preprocessor.fit_transform(X_train)
    input_dim = X_train_processed.shape[1]
    output_dim = len(np.unique(y_train))
    
    # Create the wrapper model
    model = SklearnNNWrapper(input_dim, output_dim, device)
    
    # We can't put the wrapper directly in a meaningful sklearn pipeline because the wrapper expects
    # processed features to initialize (needs input_dim). 
    # BUT, we can make the pipeline just process data, and handle fitting manually or 
    # assume the preprocessor is fixed.
    
    # A cleaner way for the pipeline object we return:
    # 1. Pipeline has preprocessor.
    # 2. We train the model manually using processed data.
    # 3. We insert the TRAINED model into the pipeline as the final step.
    
    model.fit(X_train_processed, y_train)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(report)
    
    # Save Report
    with open(os.path.join(RESULTS_DIR, f'training_report_{model_name}.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png'))
    plt.close()
    
    return y_prob

def save_artifacts(pipeline, model_name, feature_names):
    # Save the entire pipeline
    save_path = os.path.join(MODEL_DIR, f'{model_name}_pipeline.pkl')
    joblib.dump(pipeline, save_path)
    print(f"Pipeline saved to {save_path}")
    
    # Save Metadata
    meta_data = {
        "model_name": model_name,
        "feature_cols": feature_names
    }
    with open(os.path.join(MODEL_DIR, f'{model_name}_meta.json'), 'w') as f:
        json.dump(meta_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Train Loan Default Prediction Model')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'logistic', 'neural_network'], help='Model to train')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = load_data(DATA_PATH)
    
    X = data.drop(columns=['Default'])
    y = data['Default']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define Preprocessor
    preprocessor, num_cols, cat_cols = get_preprocessor(X)
    feature_names = num_cols + cat_cols # Approximate, OneHot expands this, but this is input columns
    
    if args.model == 'xgboost':
        pipeline = train_xgboost(X_train, y_train, preprocessor)
    elif args.model == 'logistic':
        pipeline = train_logistic(X_train, y_train, preprocessor)
    elif args.model == 'neural_network':
        # NN needs specific preprocessing (Normalization/StandardScaler is good)
        # We will wrap the NN model in a class that behaves like a sklearn classifier for the pipeline
        pipeline = train_neural_network(X_train, y_train, preprocessor, device)
        
    evaluate_model(pipeline, X_test, y_test, args.model)
    save_artifacts(pipeline, args.model, list(X.columns))

if __name__ == "__main__":
    main()
