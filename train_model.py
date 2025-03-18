import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load and validate dataset
    file_path = "loan_approval_dataset.csv"
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        print(f"❌ ERROR: File '{file_path}' not found!")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return

    # Clean data
    df.columns = df.columns.str.strip()
    
    # Handle target variable
    status_mapping = {'approved': 1, 'rejected': 0}
    df['loan_status'] = (df['loan_status']
                         .str.strip()
                         .str.lower()
                         .map(status_mapping)
                         .replace(r'^\s*$', np.nan, regex=True))
    
    # Clean and prepare data
    initial_count = len(df)
    df = df.dropna(subset=['loan_status'])
    df['loan_status'] = df['loan_status'].astype(int)
    print(f"🧹 Removed {initial_count - len(df)} rows with invalid loan status")

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Separate features and target
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Identify column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        X[col] = X[col].str.strip().str.lower()
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"🏷️ Encoded classes for {col}: {le.classes_}")

    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("\n✅ Model trained successfully!")

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"📊 Model accuracy: {accuracy_score(y_test, y_pred):.2%}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": X.columns.tolist(),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols
    }

    for name, obj in artifacts.items():
        with open(f"models/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
        print(f"💾 Saved {name}.pkl")

    print("\n🎉 All artifacts saved successfully!")

if __name__ == "__main__":
    main()