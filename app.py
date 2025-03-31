import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
file_path = "lab_11_bridge_data.xlsx"
df = pd.read_excel(file_path, sheet_name='lab_11_bridge_data')

# Drop Bridge_ID (not a useful feature)
df.drop(columns=['Bridge_ID'], inplace=True)

# One-hot encode the Material column
categorical_features = ['Material']
numeric_features = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Condition_Rating']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Define features and target
X = df.drop(columns=['Max_Load_Tons'])
y = df['Max_Load_Tons']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessing pipeline and transform data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save preprocessing pipeline
with open("preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

# Build ANN model
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Save trained model
model.save("tf_bridge_model.h5")

# Plot training vs validation loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
