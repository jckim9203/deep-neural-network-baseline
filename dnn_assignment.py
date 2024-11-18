import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

# Load the Titanic training dataset
data = pd.read_csv('wine_quality.csv')
data.isna().any()

# Preprocess the training data
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
features = data.columns[:-1].tolist()
target = 'quality'
target = data.columns[-1]

# Convert categorical features to numerical
data['quality'] = data['quality'].map({'low':0, 'high':1})
data.isna().any()
#train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
#train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

X = data[features].values
y = data[target].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

test_set_percentages = [0.1, 0.2]
learning_rates = [0.001, 0.01]
accuracies = []

for i in range(len(test_set_percentages)):
    for j in range(len(learning_rates)):
        # Split data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_percentages[i])

        # Build the model
        model = Sequential([
            Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # For binary classification
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rates[j]),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate the model on the validation set
        y_test_pred_prob = model.predict(X_test)  # Get predicted probabilities
        y_test_pred = (y_test_pred_prob > 0.5).astype("int32")  # Convert probabilities to binary predictions

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        class_report = classification_report(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auroc = roc_auc_score(y_test, y_test_pred_prob)

        # Print metrics
        print(f"Test set {test_set_percentages[i]*100}% learning rate {learning_rates[j]} accuracy:", accuracy)
        accuracies.append(accuracy)
        
print(accuracies)

accuracy_comparison = {
    "Test set": [0.1, 0.1, 0.2, 0.2],
    "Learning rate": [0.001, 0.01, 0.001, 0.01],
    "Accuracy": accuracies
}
accuracy_comparison_table = pd.DataFrame(accuracy_comparison)
accuracy_comparison_table
