"""GraphConvNet Model Classification module for compound activity prediction."""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.metrics import accuracy_score, f1_score
import os
import shutil
import zipfile
import matplotlib.pyplot as plt

# Helper functions
def plot_training_history(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend(loc='upper right')
    st.pyplot(plt)

def plot_true_vs_pred(y_true, y_pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    st.pyplot(plt)
import streamlit as st
import pandas as pd
import deepchem as dc
from deepchem.feat import ConvMolFeaturizer
from deepchem.models import GraphConvModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For plotting confusion matrix heatmap

# Define the ConvMolFeaturizer
featurizer = ConvMolFeaturizer()

# Function to zip a directory
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

# Function to train the model
def train_model(df, smiles_column, label_column, model_dir, batch_size, dropout, nb_epoch, graph_conv_layers, test_size, valid_size, progress_bar):
    try:
        # Featurize the SMILES column
        smiles = df[smiles_column].tolist()
        features = featurizer.featurize(smiles)

        # Extract the target values from the DataFrame
        targets = df[label_column].tolist()

        # Create a DeepChem dataset from the features and targets
        dataset = dc.data.NumpyDataset(features, targets)

        # Split the data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=test_size, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size/(1 - test_size), random_state=42)

        train_dataset = dc.data.NumpyDataset(X_train, y_train)
        valid_dataset = dc.data.NumpyDataset(X_valid, y_valid)
        test_dataset = dc.data.NumpyDataset(X_test, y_test)

        # Define and train the Graph Convolutional Model
        n_tasks = 1
        model = GraphConvModel(
            n_tasks, batch_size=batch_size, dropout=dropout, 
            graph_conv_layers=graph_conv_layers, mode='classification', 
            model_dir=model_dir
        )

        # Progress bar setup
        progress_bar.progress(0)
        progress_text = st.empty()

        training_history = {'loss': [], 'val_loss': [], 'test_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(nb_epoch):
            loss = model.fit(train_dataset, nb_epoch=1)
            training_history['loss'].append(loss)

            # Update progress bar and text
            progress = (epoch + 1) / nb_epoch
            progress_bar.progress(progress)
            progress_text.text(f"Training epoch {epoch + 1}/{nb_epoch}")

            # Calculate validation loss and accuracy
            train_scores = model.evaluate(train_dataset, [dc.metrics.Metric(dc.metrics.accuracy_score)])
            val_scores = model.evaluate(valid_dataset, [dc.metrics.Metric(dc.metrics.accuracy_score)])
            test_scores = model.evaluate(test_dataset, [dc.metrics.Metric(dc.metrics.accuracy_score)])

            training_history['accuracy'].append(train_scores['accuracy_score'])
            training_history['val_accuracy'].append(val_scores['accuracy_score'])
            training_history['val_loss'].append(val_scores['accuracy_score'])
            training_history['test_loss'].append(test_scores['accuracy_score'])

        return model, test_dataset, training_history

    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")
        return None, None, None

# Function to plot ROC AUC curve
def plot_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend(loc='upper right')
    st.pyplot(plt)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred > 0.5)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)


def main():
    st.title('Graph Convolutional Network(Classification) Modeling for Compound Acitivity Prediction')

    # Sidebar inputs
    st.sidebar.header('Model Configuration')
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file for training", type="xlsx", key="graphM_C_train_excel_1")

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.write(df.head())

        columns = df.columns.tolist()
        smiles_column = st.sidebar.selectbox("Select the SMILES column", columns)
        label_column = st.sidebar.selectbox("Select the binary label column", columns)

        # Input fields for batch size, dropout rate, and number of epochs
        batch_size = st.sidebar.text_input("Enter batch size", "256")
        dropout = st.sidebar.text_input("Enter dropout rate", "0.1")
        nb_epoch = st.sidebar.text_input("Enter number of epochs", "120")

        # Input for graph convolution layers
        graph_conv_layers = st.sidebar.text_input("Enter graph convolution layers (comma-separated)", "64,64")

        # Input fields for test size and validation size
        test_size = st.sidebar.text_input("Enter test size (fraction)", "0.15")
        valid_size = st.sidebar.text_input("Enter validation size (fraction)", "0.15")

        progress_bar = st.sidebar.progress(0)

        if st.sidebar.button("Train Model"):
            if smiles_column and label_column:
                st.write("Training the model...")
                model_dir = "./trained_model"
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.makedirs(model_dir)

                # Convert input values to appropriate types
                batch_size = int(batch_size)
                dropout = float(dropout)
                nb_epoch = int(nb_epoch)
                test_size = float(test_size)
                valid_size = float(valid_size)

                # Convert graph_conv_layers to list of integers
                graph_conv_layers = [int(layer) for layer in graph_conv_layers.split(',')]

                model, test_dataset, training_history = train_model(
                    df, smiles_column, label_column, model_dir, 
                    batch_size, dropout, nb_epoch, graph_conv_layers, test_size, valid_size, progress_bar
                )

                if model is not None:
                    st.success("Model training completed.")

                    # Evaluate the model
                    if test_dataset is not None:
                        # Evaluate on test dataset
                        y_true = np.array(test_dataset.y).ravel()
                        y_pred = model.predict(test_dataset).reshape(-1, 2)[:, 1]

                        # Compute metrics
                        roc_auc_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
                        accuracy_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
                        f1_metric = dc.metrics.Metric(dc.metrics.f1_score)
                        
                        test_scores = model.evaluate(test_dataset, [roc_auc_metric, accuracy_metric, f1_metric])

                        # Display metrics
                        st.write("Test ROC AUC Score:", test_scores[roc_auc_metric.name])
                        st.write("Test Accuracy:", test_scores[accuracy_metric.name])
                        st.write("Test F1 Score:", test_scores[f1_metric.name])

                        # Plot ROC AUC curve
                        st.write("ROC AUC Curve:")
                        plot_roc_auc(y_true, y_pred)

                        # Plot training history
                        st.write("Training History:")
                        plot_training_history(training_history)

                        # Plot confusion matrix
                        st.write("Confusion Matrix:")

def main():
    st.title('Graph Convolutional Network(Classification) Modeling for Compound Acitivity Prediction')

    # Sidebar inputs
    st.sidebar.header('Model Configuration')
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file for training", type="xlsx", key="graphM_C_train_excel_2")

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.write(df.head())

        columns = df.columns.tolist()
        smiles_column = st.sidebar.selectbox("Select the Smile column", columns)
        label_column = st.sidebar.selectbox("Select the categorical label column", columns)

        # Input fields for batch size, dropout rate, and number of epochs
        batch_size = st.sidebar.text_input("Enter batch size", "256")
        dropout = st.sidebar.text_input("Enter dropout rate", "0.1")
        nb_epoch = st.sidebar.text_input("Enter number of epochs", "120")

        # Input for graph convolution layers
        graph_conv_layers = st.sidebar.text_input("Enter graph convolution layers (comma-separated)", "64,64")

        # Input fields for test size and validation size
        test_size = st.sidebar.text_input("Enter test size (fraction)", "0.15")
        valid_size = st.sidebar.text_input("Enter validation size (fraction)", "0.15")

        progress_bar = st.sidebar.progress(0)

        if st.sidebar.button("Train Model"):
            if smiles_column and label_column:
                st.write("Training the model...")
                model_dir = "./trained_model"
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.makedirs(model_dir)

                # Convert input values to appropriate types
                batch_size = int(batch_size)
                dropout = float(dropout)
                nb_epoch = int(nb_epoch)
                test_size = float(test_size)
                valid_size = float(valid_size)

                # Convert graph_conv_layers to list of integers
                graph_conv_layers = [int(layer) for layer in graph_conv_layers.split(',')]

                model, test_dataset, training_history = train_model(
                    df, smiles_column, label_column, model_dir, 
                    batch_size, dropout, nb_epoch, graph_conv_layers, test_size, valid_size, progress_bar
                )

                if model is not None:
                    st.success("Model training completed.")

                    # Evaluate the model
                    if test_dataset is not None:
                        # Evaluate on test dataset
                        y_true = np.array(test_dataset.y).ravel()
                        y_pred = model.predict(test_dataset).ravel()

                        # Compute metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred, average='weighted')
                        
                        # Display metrics
                        st.write("Test Accuracy:", accuracy)
                        st.write("Test F1 Score:", f1)

                        # Plot training history
                        st.write("Training History:")
                        plot_training_history(training_history)

                        st.write("True vs Predicted Values:")
                        plot_true_vs_pred(y_true, y_pred)

                        # Provide download link for the trained model
                        zipf = zipfile.ZipFile('trained_model.zip', 'w', zipfile.ZIP_DEFLATED)
                        zipdir(model_dir, zipf)
                        zipf.close()

                        with open('trained_model.zip', 'rb') as f:
                            st.download_button(
                                label="Download Trained Model",
                                data=f,
                                file_name='trained_model.zip',
                                mime='application/zip'
                            )

                    else:
                        st.error("Test dataset is empty. Training may have failed.")
                else:
                    st.error("Model training failed. Please check your data and try again.")
            else:
                st.error("Please select both the Smile and categorical label columns.")


if __name__ == "__main__":
    main()
