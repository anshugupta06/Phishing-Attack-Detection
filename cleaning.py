import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf  # For setting random seed

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def run_pipeline(df):
    # Step 2: Dataset Cleaning
    print("Initial DataFrame shape:", df.shape)

    # Handling the missing values
    for column in df.columns:
        if df[column].isnull().any():
            df[column].fillna(df[column].mode()[0], inplace=True)

    # Removal of duplicate rows
    df.drop_duplicates(inplace=True)
    print("\nDataFrame shape after removing duplicates:", df.shape)

    # Identifying and handling the outliers using IQR method
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print("\nDataFrame shape after outlier handling (example):", df.shape)

    # Step 3: Splitting of features and target
    X = df.drop("Type", axis=1)
    y_original = df["Type"]

    # Step 4: Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Selection of features
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X_scaled, y_original)

    # Step 6: Splitting data into Training and Testing data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_original, test_size=0.3, random_state=42)

    # Encoding the target variable for neural networks
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_original)
    num_classes = len(np.unique(y_encoded))
    y_categorical = to_categorical(y_encoded)

    # Splitting data for neural networks
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X.values.reshape(X.shape[0], X.shape[1], 1), y_categorical, test_size=0.3, random_state=42)

    # Step 7: Defining Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),  # fix predict_proba error
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "CNN": Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(num_classes, activation='softmax')
        ]),
        "BiLSTM": Sequential([
            Bidirectional(LSTM(50, activation='relu', input_shape=(X.shape[1], 1))),
            Dense(num_classes, activation='softmax')
        ])
    }

    # Compile neural network models
    models["CNN"].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    models["BiLSTM"].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 8: Training and Evaluation
    results = {}
    val_preds = []  # store train set probabilities for stacking meta model
    test_preds = []  # store test set probabilities for stacking meta model

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name in ["CNN", "BiLSTM"]:
            model.fit(X_train_nn, y_train_nn, epochs=10, batch_size=32, verbose=0)
            y_pred_probs = model.predict(X_test_nn)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_test_true = np.argmax(y_test_nn, axis=1)
            val_preds.append(model.predict(X_train_nn))
            test_preds.append(y_pred_probs)
        else:
            model.fit(X_train, y_train)
            y_pred_probs = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            y_test_true = y_test.values
            val_preds.append(model.predict_proba(X_train))
            test_preds.append(y_pred_probs)

        results[name] = {
            "Accuracy": accuracy_score(y_test_true, y_pred),
            "Precision": precision_score(y_test_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test_true, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test_true, y_pred, average='weighted', zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test_true, y_pred)
        }

    # Prepare stacking data for meta model
    meta_X_train = np.hstack(val_preds)
    meta_X_test = np.hstack(test_preds)
    y_train_stack = label_encoder.transform(y_train)
    y_test_stack = label_encoder.transform(y_test)

    # Meta-model training
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_X_train, y_train_stack)
    y_meta_pred = meta_model.predict(meta_X_test)

    results["Ensemble"] = {
        "Accuracy": accuracy_score(y_test_stack, y_meta_pred),
        "Precision": precision_score(y_test_stack, y_meta_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test_stack, y_meta_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test_stack, y_meta_pred, average='weighted', zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test_stack, y_meta_pred)
    }

    # Step 10: Plot confusion matrices for all models
    for name, metrics in results.items():
        plt.figure(figsize=(5, 5))
        sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    # Step 11: Comparison Chart for all models
    metric_data = pd.DataFrame({
        model: {
            "Accuracy": results[model]["Accuracy"],
            "Precision": results[model]["Precision"],
            "Recall": results[model]["Recall"],
            "F1 Score": results[model]["F1 Score"]
        } for model in results
    })

    metric_data.plot(kind='bar', figsize=(14, 7))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "_main_":
    # Load the dataset only if the script is run directly
    try:
        df = pd.read_csv("Dataset.csv")
        all_results = run_pipeline(df.copy())  # Pass a copy to avoid potential modifications
        
        print("\nFinal Results Summary:")
        for name, metrics in all_results.items():
            print(f"{name}:")
            for metric, value in metrics.items():
                if metric != "Confusion Matrix":
                    print(f"  {metric}: {value:.4f}")
            print()

    except FileNotFoundError:
        print("Error: Dataset.csv not found. Please make sure the file is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
