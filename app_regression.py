"""Regression module for compound activity prediction using TPOT and DeepChem."""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import base64
import time
import deepchem as dc
import ssl
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Disable SSL verification for DeepChem downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Disable RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Dictionary of featurizers using DeepChem
Featurizer = {
    "Circular Fingerprint": dc.feat.CircularFingerprint(size=2048, radius=4),
    "MACCSKeys": dc.feat.MACCSKeysFingerprint(),
    "modred": dc.feat.MordredDescriptors(ignore_3D=True),
    "rdkit": dc.feat.RDKitDescriptors(),
    "pubchem":dc.feat.PubChemFingerprint(),
    "mol2vec":dc.feat.Mol2VecFingerprint()
}

# Function to standardize Smile using RDKit
def standardize_smiles(smiles, verbose=False):
    if verbose:
        st.write(smiles)
    std_mol = standardize_mol(Chem.MolFromSmiles(smiles), verbose=verbose)
    return Chem.MolToSmiles(std_mol)

# Function to standardize molecule using RDKit
def standardize_mol(mol, verbose=False):
    from rdkit.Chem.MolStandardize import rdMolStandardize
    
    clean_mol = rdMolStandardize.Cleanup(mol)
    if verbose:
        st.write('Remove Hs, disconnect metal atoms, normalize the molecule, reionize the molecule:')

    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    if verbose:
        st.write('Select the "parent" fragment:')

    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    if verbose:
        st.write('Neutralize the molecule:')

    te = rdMolStandardize.TautomerEnumerator()
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    if verbose:
        st.write('Enumerate tautomers:')

    assert taut_uncharged_parent_clean_mol is not None
    if verbose:
        st.write(Chem.MolToSmiles(taut_uncharged_parent_clean_mol))

    return taut_uncharged_parent_clean_mol

# Function to preprocess data and perform modeling for regression
def preprocess_and_model(df, smiles_col, activity_col, featurizer_name, generations=5, population_size=20, cv=5, test_size=0.20, random_state=42, verbosity=2):
    # Standardize Smile
    df[smiles_col + '_standardized'] = df[smiles_col].apply(standardize_smiles)
    df.dropna(subset=[smiles_col + '_standardized'], inplace=True)

    # Featurize molecules
    featurizer = Featurizer[featurizer_name]
    features = []
    for smiles in df[smiles_col + '_standardized']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features.append(featurizer.featurize([mol])[0])
        else:
            st.warning(f"Invalid Smile: {smiles}")

    if not features:
        st.error("No valid molecules found for featurization. Please ensure your Smile data is correct.")
        return None, None, None, None, None, None, None

    feature_df = pd.DataFrame(features)


    # Merge features with original dataframe
    #df = pd.concat([df, feature_df], axis=1)

    # Drop any feature containing null values
    #df.dropna(axis=1, inplace=True)
    
    #df.iloc[:, 0:]

    # TPOT modeling for classification
    X = feature_df
  
    
    y = df[activity_col]

    # Convert integer column names to strings
    new_column_names = [f"fp_{col}" for col in X.columns]
    X.columns = new_column_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tpot = TPOTRegressor(generations=generations, population_size=population_size, cv=cv, random_state=random_state, verbosity=verbosity)

    # Simulate optimization progress
    st.subheader("TPOT Optimization Progress")
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress_bar():
        for i in range(1, 121):  # Simulate 120 evaluations
            time.sleep(0.1)  # Simulate pipeline evaluation time
            progress_bar.progress(i / 120)
            progress_text.text(f"Optimization Progress: {i}/120 pipelines evaluated")

    # Uncomment the following line to run actual TPOT optimization (will take longer to complete)
    tpot.fit(X_train, y_train)

    # Dummy loop for demonstration (comment this out when using actual TPOT optimization)
    update_progress_bar()

    # Model evaluation
    y_pred = tpot.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Best TPOT Pipeline")
    st.write(tpot.fitted_pipeline_)

    # Save TPOT model and X_train separately
    with open('best_model.pkl', 'wb') as f_model:
        joblib.dump(tpot.fitted_pipeline_, f_model)
    
    with open('X_train.pkl', 'wb') as f_X_train:
        joblib.dump(X_train, f_X_train)

    # Get feature names used in modeling
    feature_names = list(X_train.columns)

    return tpot, mse, rmse, r2, y_test, y_pred, df, X_train, y_train, featurizer


# Function to interpret prediction using LIME
def interpret_prediction(tpot_model, input_features, X_train):
    # Create LIME explainer using X_train
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="regression",
        feature_names=X_train.columns,
        verbose=True
    )
    
    explanation = explainer.explain_instance(
        input_features.values[0],
        tpot_model.predict,
        num_features=len(input_features.columns)
    )

    # Generate HTML explanation
    html_explanation = explanation.as_html()
    return html_explanation


# Function to predict from single Smile input and return RDKit image
def predict_from_single_smiles(single_smiles, featurizer_name='Circular Fingerprint'):
    standardized_smiles = standardize_smiles(single_smiles, verbose=False)
    if standardized_smiles:
        mol = Chem.MolFromSmiles(standardized_smiles)
        if mol is not None:
            featurizer = Featurizer[featurizer_name]
            features = featurizer.featurize([mol])[0]
            input_features = pd.DataFrame([features])

            # Convert integer column names to strings
            new_column_names = [f"fp_{col}" for col in input_features.columns]
            input_features.columns = new_column_names

            # Load trained model and X_train
            try:
                with open('best_model.pkl', 'rb') as f_model, open('X_train.pkl', 'rb') as f_X_train:
                    tpot_model = joblib.load(f_model)
                    X_train = joblib.load(f_X_train)
            except FileNotFoundError:
                st.warning("Please build and save the model in the 'Build Model' section first.")
                return None, None, None, None

            # Predict using the trained model
            prediction = tpot_model.predict(input_features)[0]

            # Interpret prediction using LIME
            explanation_html = interpret_prediction(tpot_model, input_features, X_train)

            # Generate RDKit image of the molecule
            img = Chem.Draw.MolToImage(mol, size=(300, 300))

            return prediction, explanation_html, img
        else:
            st.warning("Invalid Smile input. Please check your input and try again.")
            return None, None, None
    else:
        st.warning("Invalid Smile input. Please check your input and try again.")
        return None, None, None


# Function to create a downloadable link for HTML content
def create_download_link(html_content, link_text):
    # Create HTML link
    href = f'<a href="data:text/html;base64,{base64.b64encode(html_content.encode()).decode()}" download="explanation.html">{link_text}</a>'
    return href


# Function to plot predicted vs true values
def plot_predicted_vs_true(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    st.pyplot(fig)


# Main function to run the Streamlit app
def main():
    st.title("Chemical Activity Prediction (Regression)")

    # Initialize session state (this dictionary will hold the session-specific state)
    if 'selected_featurizer_name' not in st.session_state:
        st.session_state.selected_featurizer_name = "Circular Fingerprint"

    # Navigation
    options = ["Home", "Build Model", "Predict from Smile", "Predict from Excel"]
    choice = st.sidebar.selectbox("Choose a task", options)

    if choice == "Home":
        st.subheader("About This App")
        st.markdown("""
        **Chemical Activity Prediction (Regression)**
        
        This app allows you to:
        - Build regression models for chemical activity prediction using TPOT and DeepChem featurizers.
        - Predict activity values from single SMILES or batch Excel files.
        - Visualize model performance and interpret results.
        
        **How to use:**
        1. Go to 'Build Model' to upload your data and train a regression model.
        2. Use 'Predict from Smile' for single predictions.
        3. Use 'Predict from Uploaded Excel File' for batch predictions.
        
        Select an option from the sidebar to get started!
        """)

    elif choice == "Build Model":
        st.subheader("Build Model")
        st.write("Upload an Excel file containing Smile and corresponding activity values to train the model.")
        uploaded_file = st.file_uploader("Upload Excel file with Smile and Activity", type=["xlsx"], key="app_regression_train_excel")

        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Show dataframe
            st.subheader("Uploaded Data for Model Building")
            st.write(df)

            # Sidebar options for selecting Smile and Activity columns
            with st.sidebar:
                # Select Smile and Activity columns
                col_names = df.columns.tolist()
                smiles_col = st.selectbox("Select Smile Column", col_names, key='smiles_column')
                activity_col = st.selectbox("Select Activity Column", col_names, key='activity_column')

                # Select Featurizer
                st.session_state.selected_featurizer_name = st.selectbox("Select Featurizer", list(Featurizer.keys()), key='featurizer_name', index=list(Featurizer.keys()).index(st.session_state.selected_featurizer_name))

                # Input fields for TPOT parameters
                generations = st.slider("Generations", min_value=1, max_value=50, value=5)
                cv = st.slider("Cross-validation folds", min_value=2, max_value=50, value=5)
                verbosity = st.slider("Verbosity", min_value=0, max_value=2, value=2)
                test_size = st.slider("Test Size", min_value=0.05, max_value=0.50, value=0.20, step=0.05)

            # Build and train the model
            if st.button("Build and Train Model"):
                st.write("Building and training the model...")
                tpot, mse, rmse, r2, y_test, y_pred, df, X_train, y_train, featurizer = preprocess_and_model(
                    df, smiles_col, activity_col, st.session_state.selected_featurizer_name, generations, cv=cv, test_size=test_size, verbosity=verbosity
                )

                # Display model metrics
                st.subheader("Model Evaluation Metrics")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                st.write(f"R-squared (R2): {r2:.2f}")

                # Plot predicted vs true values
                plot_predicted_vs_true(y_test, y_pred)

    elif choice == "Predict from Smile":
        st.subheader("Predict from Smile")
        smile_input = st.text_input("Enter Smile for prediction")

        if st.button("Predict from Smile"):
            if smile_input:
                prediction, explanation_html, img = predict_from_single_smiles(smile_input, st.session_state.selected_featurizer_name)
                if prediction is not None:
                    st.write(f"Predicted Activity Value: {prediction:.4f}")

                    # Display RDKit image and LIME explanation as HTML
                    st.image(img, caption='Chemical Structure', width=300)
                    st.markdown(create_download_link(explanation_html, "Explanation"), unsafe_allow_html=True)
                else:
                    st.warning("Failed to make prediction. Please check your input and try again.")

    elif choice == "Predict from Excel":
        st.subheader("Predict from Excel")
        uploaded_pred_file = st.file_uploader("Upload Excel file with Smile for prediction", type=["xlsx"], key="app_regression_pred_excel")

        if uploaded_pred_file is not None:
            # Read Excel file for prediction
            pred_df = pd.read_excel(uploaded_pred_file)

            # Show dataframe for prediction
            st.subheader("Uploaded Data for Prediction")
            st.write(pred_df)

            # Sidebar options for selecting Smile column for prediction
            with st.sidebar:
                # Select Smile column for prediction
                pred_col_names = pred_df.columns.tolist()
                pred_smiles_col = st.selectbox("Select Smile Column for Prediction", pred_col_names, key='pred_smiles_column')

            # Button to trigger prediction
            if st.button("Predict from Excel"):
                predictions = []
                best_pipeline_info = ""

                for idx, smiles in enumerate(pred_df[pred_smiles_col]):
                    st.write(f"Predicting for Smile {idx + 1}: {smiles}")
                    prediction, explanation_html, img = predict_from_single_smiles(smiles, st.session_state.selected_featurizer_name)
                    
                    if prediction is not None:
                        predictions.append(prediction)
                        # Display RDKit image and LIME explanation as HTML
                        st.image(img, caption=f'Chemical Structure for Smile {idx + 1}', width=300)
                        st.markdown(create_download_link(explanation_html, f"Explanation_{idx + 1}"), unsafe_allow_html=True)
                        st.write(f"Predicted Activity Value: {prediction:.4f}")  # Display predicted value
                    else:
                        predictions.append("Failed")
                
                # Show predictions
                st.subheader("Predictions")
                pred_df['Predicted Activity'] = predictions
                st.write(pred_df)               

if __name__ == "__main__":
    main()
