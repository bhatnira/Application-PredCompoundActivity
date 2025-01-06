import streamlit as st
from zipfile import ZipFile
import os
import shutil
from rdkit import Chem
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps

# Function to extract uploaded zip file
def extract_zip(zip_file):
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("temp_model_dir")
    return "temp_model_dir"

# Function to load the model from the extracted directory
def load_model(model_dir):
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' does not exist.")

    try:
        n_tasks = 1  # Assuming 1 task for simplicity
        model = dc.models.GraphConvModel(n_tasks, model_dir=model_dir)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model from '{model_dir}': {str(e)}")

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

# Function to make predictions for whole molecules
def predict_whole_molecules(model, dataset):
    try:
        predictions = np.squeeze(model.predict(dataset))
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)
        predictions_df = pd.DataFrame(predictions[:, 1], index=dataset.ids, columns=["Probability_Class_1"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for fragments
def predict_fragment_dataset(model, frag_dataset):
    try:
        predictions = np.squeeze(model.predict(frag_dataset))[:, 1]
        predictions_df = pd.DataFrame(predictions, index=frag_dataset.ids, columns=["Fragment"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

# Function to visualize contributions
def vis_contribs(mols, df, smi_or_sdf="sdf"):
    maps = []
    for mol  in mols:
        wt = {}
        if smi_or_sdf == "smi":
            for n,atom in enumerate(Chem.rdmolfiles.CanonicalRankAtoms(mol)):
                wt[atom] = df.loc[mol.GetProp("_Name"),"Contrib"][n]
        if smi_or_sdf == "sdf":
            for n,atom in enumerate(range(mol.GetNumHeavyAtoms())):
                wt[atom] = df.loc[Chem.MolToSmiles(mol),"Contrib"][n]
        maps.append(SimilarityMaps.GetSimilarityMapFromWeights(mol,wt))
    return maps

# Streamlit app
st.title('Interpretable Acivity Prediction(deployment)-GraphConv Classification')

# File upload widget for model zip file
uploaded_file = st.file_uploader("Upload model zip file", type="zip")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.read())

    # Extract the uploaded zip file
    extracted_dir = extract_zip("temp.zip")

    # Load the model from the extracted directory
    model_dir = extracted_dir  # Assuming the extracted directory is where the model files are
    model = load_model(model_dir)

    st.success("Model loaded successfully!")

    # Input type selection: Smile or Excel File
    input_type = st.radio('Choose input type:', ('Smile', 'Excel File'))

    if input_type == 'Smile':
        # Input Smile
        input_smiles = st.text_input('Enter Smile string')

        if input_smiles:
            # Convert Smile to SDF
            sdf_path = "input_molecule.sdf"
            sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

            if error:
                st.error(f"Error in Smile to SDF conversion: {error}")
            else:
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
                                df['Contrib'] = df["Probability_Class_1"] - df["Fragment"]

                                # Generate molecule from input Smile
                                mol = Chem.MolFromSmiles(input_smiles)

                                # Create maps for the molecule
                                if mol:
                                    maps = vis_contribs([mol], df)
                                    st.write(f"Contribution Map 1:")
                                    st.write(maps[0])

                                    # Binary prediction (active/inactive)
                                    threshold = 0.5
                                    prediction_binary = "Active" if predictions_whole.iloc[0, 0] > threshold else "Inactive"
                                    st.write(f"Binary Prediction: {prediction_binary}")

                                    # Prediction probability for class 1
                                    st.write(f"Prediction Probability for Class 1: {predictions_whole.iloc[0, 0]}")
                                else:
                                    st.warning("Unable to generate molecule from input Smile.")

                                # Clean up the SDF file after prediction
                                if os.path.exists(sdf_path):
                                    os.remove(sdf_path)

    elif input_type == 'Excel File':
        # Upload Excel file
        excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

        if excel_file is not None:
            # Load Excel file into pandas DataFrame
            try:
                df = pd.read_excel(excel_file, engine='openpyxl')
                st.write('Uploaded DataFrame:', df)

                # Select column containing Smile
                smiles_column = st.selectbox('Select column containing Smile', df.columns)

                # Initialize predictions list
                predictions_list = []

                # Iterate over each Smile and perform predictions
                for smiles in df[smiles_column]:
                    sdf_path = "input_molecule.sdf"
                    sdf_path, error = smiles_to_sdf(smiles, sdf_path)

                    if error:
                        predictions_list.append({"Smile": smiles, "Prediction": "Error: " + error})
                    else:
                        # Create dataset
                        dataset, error = create_dataset(sdf_path)

                        if error:
                            predictions_list.append({"Smile": smiles, "Prediction": "Error: " + error})
                        else:
                            # Make predictions for whole molecules
                            predictions_whole, error = predict_whole_molecules(model, dataset)

                            if error:
                                predictions_list.append({"Smile": smiles, "Prediction": "Error: " + error})
                            else:
                                # Create fragment dataset
                                frag_dataset, error = create_fragment_dataset(sdf_path)

                                if error:
                                    predictions_list.append({"Smile": smiles, "Prediction": "Error: " + error})
                                else:
                                    # Make predictions for fragments
                                    predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                                    if error:
                                        predictions_list.append({"Smile": smiles, "Prediction": "Error: " + error})
                                    else:
                                        # Merge two DataFrames by molecule names
                                        df_temp = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                                        df_temp['Contrib'] = df_temp["Probability_Class_1"] - df_temp["Fragment"]

                                        # Generate molecule from input Smile
                                        mol = Chem.MolFromSmiles(smiles)

                                        # Create maps for the molecule
                                        if mol:
                                            maps = vis_contribs([mol], df_temp)
                                            st.write(f"Contribution Map 1 for Smile: {smiles}")
                                            st.write(maps[0])

                                            # Binary prediction (active/inactive)
                                            threshold = 0.5
                                            prediction_binary = "Active" if predictions_whole.iloc[0, 0] > threshold else "Inactive"
                                            st.write(f"Binary Prediction: {prediction_binary}")

                                            # Prediction probability for class 1
                                            st.write(f"Prediction Probability for Class 1: {predictions_whole.iloc[0, 0]}")
                                        else:
                                            st.warning(f"Unable to generate molecule from Smile: {smiles}")

                    # Append predictions to list
                    predictions_list.append({
                        "Smile": smiles,
                        "Binary Prediction": prediction_binary,
                        "Probability_Class_1": predictions_whole.iloc[0, 0]
                    })

                    # Clean up the SDF file after prediction
                    if os.path.exists(sdf_path):
                        os.remove(sdf_path)

                # Display predictions in a DataFrame
                predictions_df = pd.DataFrame(predictions_list)
                st.write('Predictions:')
                st.write(predictions_df[["Smile", "Binary Prediction", "Probability_Class_1"]])

            except Exception as e:
                st.error(f"Error: {e}")

    # Clean up extracted directory
    shutil.rmtree(extracted_dir, ignore_errors=True)
