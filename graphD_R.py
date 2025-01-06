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

# Streamlit app
st.title('Interpretable Acivity Prediction(deployment)-GraphConv Regression')

# Handle model zip file upload
uploaded_zip = st.file_uploader("Upload model zip file", type=['zip'])

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
def create_fragment_dataset(sdf_path):
    try:
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True), sanitize=True)
        frag_dataset = loader.create_dataset(sdf_path, shard_size=5000)
        transformer = dc.trans.FlatteningTransformer(frag_dataset)
        frag_dataset = transformer.transform(frag_dataset)
        return frag_dataset, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for whole molecules (regression)
def predict_whole_molecules(model, dataset):
    try:
        pred = model.predict(dataset)
        
        # Ensure pred has the correct shape for regression
        if pred.ndim == 3 and pred.shape[-1] == 2:
            pred = pred[:, :, 1]  # Assuming you want the second dimension (probability of class 1)
        
        pred = pd.DataFrame(pred, index=dataset.ids, columns=["Molecule"])  # Convert to DataFrame
        return pred, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for fragments (regression)
def predict_fragment_dataset(model, frag_dataset):
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
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
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
