"""GraphConvNet Regression module for compound activity prediction."""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.metrics import mean_squared_error, r2_score
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

# Dummy implementations for undefined functions (replace with actual logic if needed)
def predict_whole_molecules(model, dataset):
    # TODO: Implement actual prediction logic
    return pd.DataFrame({'Molecule': [0.0]}, index=['dummy']), None

def create_fragment_dataset(sdf_path):
    # TODO: Implement actual fragment dataset logic
    return pd.DataFrame({'Fragment': [0.0]}, index=['dummy']), None

def predict_fragment_dataset(model, frag_dataset):
    # TODO: Implement actual fragment prediction logic
    return pd.DataFrame({'Fragment': [0.0]}, index=['dummy']), None

# Function to zip a directory (stub)
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))
import streamlit as st
import os
import zipfile
import io
from rdkit import Chem
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit.Chem.Draw import SimilarityMaps
import tensorflow as tf
tf.random.set_seed(42)

# Function to convert Smile to SDF
def smiles_to_sdf(smiles, sdf_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid Smile string"
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    return sdf_path, None

# Function to create dataset
def create_dataset(sdf_path):
    try:
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(), sanitize=True)
        dataset = loader.create_dataset(sdf_path, shard_size=2000)
        return dataset, None
    except Exception as e:
        return None, str(e)

# Function to create fragment dataset

def main():
    st.title('Graph Convolutional Network(Regression) Modeling for Compound Acitivity Prediction')

    # Sidebar inputs
    st.sidebar.header('Model Configuration')
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file for training", type="xlsx", key="graphD_R_train_excel")

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.write(df.head())

        columns = df.columns.tolist()
        smiles_column = st.sidebar.selectbox("Select the Smile column", columns)
        label_column = st.sidebar.selectbox("Select the continuous label column", columns)

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
                        mse = mean_squared_error(y_true, y_pred)
                        r2 = r2_score(y_true, y_pred)
                        
                        # Display metrics
                        st.write("Test Mean Squared Error:", mse)
                        st.write("Test R² Score:", r2)

                        # Plot training history
                        st.write("Training History:")
                        plot_training_history(training_history)

                        st.write("True vs Predicted Values:")
## Removed duplicate main() and training logic for clarity
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

def main():
    st.title('Interpretable Acivity Prediction(deployment)-GraphConv Regression')
    # Handle model zip file upload
    uploaded_zip = st.file_uploader("Upload model zip file", type=['zip'], key="graphD_R_model_zip_1")

    model_dir = None
    if uploaded_zip is not None:
        try:
            # Create a temporary directory to extract the zip file
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                model_dir = 'temp_model_dir'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                zip_ref.extractall(model_dir)
            st.success("Model zip file uploaded and extracted successfully!")
        except Exception as e:
            st.error(f"Error extracting zip file: {str(e)}")
            st.stop()

    # Check if model directory exists
    if model_dir and not os.path.exists(model_dir):
        st.error(f"Model directory '{model_dir}' does not exist.")
        st.stop()

    try:
        # Load the model
        n_tasks = 1  # Assuming 1 task for simplicity
        model = dc.models.GraphConvModel(n_tasks, model_dir=model_dir)
        model.restore()
        st.success("Model loaded successfully!")
    except Exception as e:
        pass
       # st.error(f"Error loading model from '{model_dir}': {str(e)}")


    # Input type selection: Smile or Excel File
    input_type = st.radio('Choose input type:', ('Smile', 'Excel File'))

    if input_type == 'Smile':
        # Input Smile
        input_smiles = st.text_input('Enter Smile string')

        if st.button('Predict and Show Contribution Map'):
            if input_smiles:
                # Convert Smile to SDF
                sdf_path = "input_molecule.sdf"
                sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

                if error:
                    st.error(f"Error in Smile to SDF conversion: {error}")
                else:
                    try:
                        # Create dataset
                        dataset, error = create_dataset(sdf_path)

                        if error:
                            st.error(f"Error in dataset creation: {error}")
                        else:
                            # Make predictions for whole molecules
                            predictions_whole, error = predict_whole_molecules(model, dataset)

                            if error:
                                st.error(f"Error in predicting whole molecules: {error}")
                            else:
                                # Create fragment dataset
                                frag_dataset, error = create_fragment_dataset(sdf_path)

                                if error:
                                    st.error(f"Error in fragment dataset creation: {error}")
                                else:
                                    # Make predictions for fragments
                                    predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                                    if error:
                                        st.error(f"Error in predicting fragments: {error}")
                                    else:
                                        # Merge two DataFrames by molecule names
                                        df = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                                        df['Contrib'] = df["Molecule"] - df["Fragment"]

                                        # Generate molecule from input Smile
                                        mol = Chem.MolFromSmiles(input_smiles)

                                        # Create maps for the molecule
                                        if mol:
                                            contribution_map = vis_contribs(mol, df)
                                            if contribution_map:
                                                st.write(f"Predicted Value: {(df['Molecule'].iloc[0])}")
                                                st.write(f"Contribution Map:")
                                                st.write(contribution_map)
                                            else:
                                                st.warning("Error in generating contribution map.")
                                        else:
                                            st.warning("Unable to generate molecule from input Smile.")

                                        # Clean up the SDF file after prediction
                                        if os.path.exists(sdf_path):
                                            os.remove(sdf_path)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    elif input_type == 'Excel File':
        # Handle Excel file input
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'], key="graphD_R_excel_1")
        if uploaded_file is not None:
            try:
                # Load Excel file
                df = pd.read_excel(uploaded_file)

                # Predict for each Smile in the Excel file
                st.write("Predicting from uploaded Excel file...")
                for index, row in df.iterrows():
                    input_smiles = row['Smile']  # Assuming 'Smile' is the column name

                    # Convert Smile to SDF
                    sdf_path = f"input_molecule_{index}.sdf"
                    sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

                    if error:
                        st.error(f"Error in Smile to SDF conversion for row {index}: {error}")
                        continue

                    # Create dataset
                    dataset, error = create_dataset(sdf_path)

                    if error:
                        st.error(f"Error in dataset creation for row {index}: {error}")
                        continue

                    # Make predictions for whole molecules
                    predictions_whole, error = predict_whole_molecules(model, dataset)

                    if error:
                        st.error(f"Error in predicting whole molecules for row {index}: {error}")
                        continue

                    # Create fragment dataset
                    frag_dataset, error = create_fragment_dataset(sdf_path)

                    if error:
                        st.error(f"Error in fragment dataset creation for row {index}: {error}")
                        continue

                    # Make predictions for fragments
                    predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                    if error:
                        st.error(f"Error in predicting fragments for row {index}: {error}")
                        continue

                    # Merge two DataFrames by molecule names
                    df_result = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                    df_result['Contrib'] = df_result["Molecule"] - df_result["Fragment"]

                    # Generate molecule from input Smile
                    mol = Chem.MolFromSmiles(input_smiles)

                    # Create maps for the molecule
                    if mol:
                        contribution_map = vis_contribs(mol, df_result)
                        if contribution_map:
                            st.write(f"------------------------------------------------------------")
                            st.write(f"Prediction and contrib map for input: {input_smiles}:")
                            st.write(f"Predicted Activity(nM): {(df_result['Molecule'].iloc[0])}")
                            st.write(contribution_map)
                        else:
                            st.warning(f"Error in generating contribution map for Smile {input_smiles}.")
                    else:
                        st.warning(f"Unable to generate molecule from input Smile {input_smiles}.")

                    # Clean up the SDF file after prediction
                    if os.path.exists(sdf_path):
                        os.remove(sdf_path)

            except Exception as e:
                st.error(f"Error processing uploaded Excel file: {str(e)}")
    try:
        pred_frags = model.predict(frag_dataset)
        
        # Ensure pred_frags has the correct shape for regression
        if pred_frags.ndim == 3 and pred_frags.shape[-1] == 2:
            pred_frags = pred_frags[:, :, 1]  # Assuming you want the second dimension (probability of class 1)
        
        pred_frags = pd.DataFrame(pred_frags, index=frag_dataset.ids, columns=["Fragment"])  # Convert to DataFrame
        return pred_frags, None
    except Exception as e:
        return None, str(e)

# Function to visualize contributions
def vis_contribs(mol, df):
    try:
        wt = {}
        for n, atom in enumerate(range(mol.GetNumHeavyAtoms())):
            wt[atom] = df.loc[Chem.MolToSmiles(mol), "Contrib"][n]
        return SimilarityMaps.GetSimilarityMapFromWeights(mol, wt)
    except Exception as e:
        st.warning(f"Error in contribution visualization: {str(e)}")
        return None


def main():
    st.title('Interpretable Acivity Prediction(deployment)-GraphConv Regression')
    # Handle model zip file upload
    uploaded_zip = st.file_uploader("Upload model zip file", type=['zip'], key="graphD_R_model_zip_2")

    model_dir = None
    if uploaded_zip is not None:
        try:
            # Create a temporary directory to extract the zip file
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                model_dir = 'temp_model_dir'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                zip_ref.extractall(model_dir)
            st.success("Model zip file uploaded and extracted successfully!")
        except Exception as e:
            st.error(f"Error extracting zip file: {str(e)}")
            st.stop()

    # Check if model directory exists
    if model_dir and not os.path.exists(model_dir):
        st.error(f"Model directory '{model_dir}' does not exist.")
        st.stop()

    try:
        # Load the model
        n_tasks = 1  # Assuming 1 task for simplicity
        model = dc.models.GraphConvModel(n_tasks, model_dir=model_dir)
        model.restore()
        st.success("Model loaded successfully!")
    except Exception as e:
        pass
       # st.error(f"Error loading model from '{model_dir}': {str(e)}")

    # Input type selection: Smile or Excel File
    input_type = st.radio('Choose input type:', ('Smile', 'Excel File'))

    if input_type == 'Smile':
        # Input Smile
        input_smiles = st.text_input('Enter Smile string')

        if st.button('Predict and Show Contribution Map'):
            if input_smiles:
                # Convert Smile to SDF
                sdf_path = "input_molecule.sdf"
                sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

                if error:
                    st.error(f"Error in Smile to SDF conversion: {error}")
                else:
                    try:
                        # Create dataset
                        dataset, error = create_dataset(sdf_path)

                        if error:
                            st.error(f"Error in dataset creation: {error}")
                        else:
                            # Make predictions for whole molecules
                            predictions_whole, error = predict_whole_molecules(model, dataset)

                            if error:
                                st.error(f"Error in predicting whole molecules: {error}")
                            else:
                                # Create fragment dataset
                                frag_dataset, error = create_fragment_dataset(sdf_path)

                                if error:
                                    st.error(f"Error in fragment dataset creation: {error}")
                                else:
                                    # Make predictions for fragments
                                    predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                                    if error:
                                        st.error(f"Error in predicting fragments: {error}")
                                    else:
                                        # Merge two DataFrames by molecule names
                                        df = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                                        df['Contrib'] = df["Molecule"] - df["Fragment"]

                                        # Generate molecule from input Smile
                                        mol = Chem.MolFromSmiles(input_smiles)

                                        # Create maps for the molecule
                                        if mol:
                                            contribution_map = vis_contribs(mol, df)
                                            if contribution_map:
                                                st.write(f"Predicted Value: {(df['Molecule'].iloc[0])}")
                                                st.write(f"Contribution Map:")
                                                st.write(contribution_map)
                                            else:
                                                st.warning("Error in generating contribution map.")
                                        else:
                                            st.warning("Unable to generate molecule from input Smile.")

                                        # Clean up the SDF file after prediction
                                        if os.path.exists(sdf_path):
                                            os.remove(sdf_path)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    elif input_type == 'Excel File':
        # Handle Excel file input
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'], key="graphD_R_excel_2")
        if uploaded_file is not None:
            try:
                # Load Excel file
                df = pd.read_excel(uploaded_file)

                # Predict for each Smile in the Excel file
                st.write("Predicting from uploaded Excel file...")
                for index, row in df.iterrows():
                    input_smiles = row['Smile']  # Assuming 'Smile' is the column name

                    # Convert Smile to SDF
                    sdf_path = f"input_molecule_{index}.sdf"
                    sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

                    if error:
                        st.error(f"Error in Smile to SDF conversion for row {index}: {error}")
                        continue

                    # Create dataset
                    dataset, error = create_dataset(sdf_path)

                    if error:
                        st.error(f"Error in dataset creation for row {index}: {error}")
                        continue

                    # Make predictions for whole molecules
                    predictions_whole, error = predict_whole_molecules(model, dataset)

                    if error:
                        st.error(f"Error in predicting whole molecules for row {index}: {error}")
                        continue

                    # Create fragment dataset
                    frag_dataset, error = create_fragment_dataset(sdf_path)

                    if error:
                        st.error(f"Error in fragment dataset creation for row {index}: {error}")
                        continue

                    # Make predictions for fragments
                    predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                    if error:
                        st.error(f"Error in predicting fragments for row {index}: {error}")
                        continue

                    # Merge two DataFrames by molecule names
                    df_result = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                    df_result['Contrib'] = df_result["Molecule"] - df_result["Fragment"]

                    # Generate molecule from input Smile
                    mol = Chem.MolFromSmiles(input_smiles)

                    # Create maps for the molecule
                    if mol:
                        contribution_map = vis_contribs(mol, df_result)
                        if contribution_map:
                            st.write(f"------------------------------------------------------------")
                            st.write(f"Prediction and contrib map for input: {input_smiles}:")
                            st.write(f"Predicted Activity(nM): {(df_result['Molecule'].iloc[0])}")
                            st.write(contribution_map)
                        else:
                            st.warning(f"Error in generating contribution map for Smile {input_smiles}.")
                    else:
                        st.warning(f"Unable to generate molecule from input Smile {input_smiles}.")

                    # Clean up the SDF file after prediction
                    if os.path.exists(sdf_path):
                        os.remove(sdf_path)

            except Exception as e:
                st.error(f"Error processing uploaded Excel file: {str(e)}")
