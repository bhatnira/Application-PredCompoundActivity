import streamlit as st
import pandas as pd
from rdkit import Chem
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import base64
import matplotlib.pyplot as plt
import time
import ssl
import deepchem as dc
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

ssl._create_default_https_context = ssl._create_unverified_context

# Dictionary of featurizers using DeepChem
Featurizer = {
    "Circular Fingerprint": dc.feat.CircularFingerprint(size=2048, radius=4),
    "MACCSKeys": dc.feat.MACCSKeysFingerprint(),
    "modred": dc.feat.MordredDescriptors(ignore_3D=True),
    "rdkit": dc.feat.RDKitDescriptors()
}

# Function to standardize SMILES using RDKit
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
        # draw_mol_with_SVG(clean_mol)

    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    if verbose:
        st.write('Select the "parent" fragment:')
        # draw_mol_with_SVG(parent_clean_mol)

    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    if verbose:
        st.write('Neutralize the molecule:')
        # draw_mol_with_SVG(uncharged_parent_clean_mol)

    te = rdMolStandardize.TautomerEnumerator()
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    if verbose:
        st.write('Enumerate tautomers:')
        # draw_mol_with_SVG(taut_uncharged_parent_clean_mol)

    assert taut_uncharged_parent_clean_mol is not None
    if verbose:
        st.write(Chem.MolToSmiles(taut_uncharged_parent_clean_mol))

    return taut_uncharged_parent_clean_mol

# Function to preprocess data and perform modeling for classification
def preprocess_and_model(df, smiles_col, activity_col, featurizer_name, generations=5, cv=5, verbosity=2):
    # Standardize SMILES
    df[smiles_col + '_standardized'] = df[smiles_col].apply(standardize_smiles)
    df.dropna(subset=[smiles_col + '_standardized'], inplace=True)

    # Convert activity column to binary labels
    unique_classes = df[activity_col].unique()
    if len(unique_classes) < 2:
        st.error("Not enough classes present for binary classification. Please check your dataset and ensure it has at least two distinct classes.")
        return None, None, None, None, None, None, None, None, None, None

    # Featurize molecules
    featurizer = Featurizer[featurizer_name]
    features = []
    for smiles in df[smiles_col + '_standardized']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features.append(featurizer.featurize([mol])[0])
        else:
            st.warning(f"Invalid SMILES: {smiles}")

    if not features:
        st.error("No valid molecules found for featurization. Please ensure your SMILES data is correct.")
        return None, None, None, None, None, None, None, None, None, None

    feature_df = pd.DataFrame(features)

    # Merge features with original dataframe
    df = pd.concat([df, feature_df], axis=1)

    # Drop any feature containing null values
    df.dropna(axis=1, inplace=True)

    # TPOT modeling for classification
    X = df.drop(columns=[smiles_col, smiles_col + '_standardized', activity_col])
    y = df[activity_col]

    # Convert integer column names to strings
    new_column_names = [f"fp_{col}" for col in X.columns]
    X.columns = new_column_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    tpot = TPOTClassifier(generations=generations, population_size=20, cv=cv, random_state=42, verbosity=verbosity)

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
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC AUC score and curve
    roc_auc = None
    if hasattr(tpot, 'predict_proba'):
        y_proba = tpot.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # Plot ROC curve
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend(loc='lower right')
        st.subheader("ROC Curve")
        st.pyplot(fig_roc)
    else:
        st.warning("ROC AUC score could not be calculated due to insufficient classes.")

    # Confusion Matrix Heatmap
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    st.subheader("Confusion Matrix Heatmap")
    st.pyplot()

    # Print best pipeline
    st.subheader("Best TPOT Pipeline")
    st.write(tpot.fitted_pipeline_)

    # Save TPOT model and X_train separately
    with open('best_model.pkl', 'wb') as f_model:
        joblib.dump(tpot.fitted_pipeline_, f_model)
    
    with open('X_train.pkl', 'wb') as f_X_train:
        joblib.dump(X_train, f_X_train)

    # Get feature names used in modeling
    feature_names = list(X_train.columns)

    return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer


# Function to interpret prediction using LIME
def interpret_prediction(tpot_model, input_features, X_train):
    # Load model for prediction
    model_fn = eval_model(tpot_model)

    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="classification",
        feature_names=input_features.columns,
        class_names=["Not Active", "Active"],
        discretize_continuous=True
    )

    # Extract fragments from the input molecule
    smiles_input = input_features.values[0]
    mol = Chem.MolFromSmiles(smiles_input)
    my_fragments = fp_mol(mol)

    # Explain prediction
    explanation = explainer.explain_instance(
        input_features.values[0],
        model_fn,
        num_features=len(input_features.columns),
        top_labels=1
    )

    # Convert explanation to a dictionary mapping fingerprint indices to weights
    fragment_weight = dict(explanation.as_map()[1])

    # Print fragments that contributed to the prediction
    for index in my_fragments:
        if index in fragment_weight:
            print(index, my_fragments[index], fragment_weight[index])

    # Generate HTML explanation for display in Streamlit
    html_explanation = explanation.as_html(show_table=True, show_all=False)
    return html_explanation


# Function to predict from SMILES input
def predict_from_smiles(tpot_model, X_train, smiles):
    mol = Chem.MolFromSmiles(smiles)
    features = featurizer.featurize([mol])[0]
    input_features = pd.DataFrame([features], columns=X_train.columns)
    prediction = tpot_model.predict(input_features)[0]
    return prediction


# Streamlit application code
def main():
    st.title("Chemical Activity Prediction")
    st.sidebar.title("Menu")
    choice = st.sidebar.radio("Navigation", ("Optimize Parameters", "Import Data", "Train Model", "Predict from SMILES"))

    # Load or import data
    if choice == "Optimize Parameters":
        st.subheader("Optimize Parameters")
        st.write("This section will perform parameter optimization using TPOT.")

        # Placeholder for optimization controls

    elif choice == "Import Data":
        st.subheader("Import Data")
        st.write("This section will allow you to upload your dataset.")

        # Placeholder for dataset upload and preview

    elif choice == "Train Model":
        st.subheader("Train Model")
        st.write("This section will train a model using the selected featurization method and dataset.")

        # Placeholder for training controls
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            # Perform modeling
            featurizer_name = st.selectbox("Select Featurization Method", list(Featurizer.keys()))
            generations = st.slider("Select TPOT Generations", min_value=1, max_value=10, value=5)
            tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer = preprocess_and_model(df, 'smiles', 'activity', featurizer_name, generations)
            st.write(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
            
            # Perform interpretation on a sample molecule
            if st.button("Interpret Sample Prediction"):
                sample_smiles = df.iloc[0]['smiles']
                st.write(f"Interpreting prediction for molecule: {sample_smiles}")
                html_explanation = interpret_prediction(tpot, pd.DataFrame([sample_smiles], columns=['smiles']), X_train)
                st.write(html_explanation, unsafe_allow_html=True)

    elif choice == "Predict from SMILES":
        st.subheader("Predict from SMILES")
        smile_input = st.text_input("Enter SMILES for prediction")

        if st.button("Predict"):
            st.write("Predicting...")
            tpot_model = joblib.load('best_model.pkl')
            X_train = joblib.load('X_train.pkl')

            prediction = predict_from_smiles(tpot_model, X_train, smile_input)

            st.subheader("Prediction")
            if prediction == 1:
                st.write("The molecule is predicted to be Active.")
            else:
                st.write("The molecule is predicted to be Not Active.")

            # Interpret prediction using LIME
            st.subheader("Interpreting Prediction using LIME")
            st.write("This section explains why the molecule was predicted as Active or Not Active.")

            # Perform interpretation
            html_explanation = interpret_prediction(tpot_model, pd.DataFrame([smile_input], columns=['smiles']), X_train)
            st.write(html_explanation, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
