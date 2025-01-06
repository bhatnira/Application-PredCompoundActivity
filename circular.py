import streamlit as st
import pandas as pd
from rdkit import Chem
import deepchem as dc
from lime import lime_tabular
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import base64
import matplotlib.pyplot as plt
import time
import ssl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Disable SSL certificate verification warning
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

    # LIME Explainer
    explainer = lime_tabular.LimeTabularExplainer(X_train.values,
                                                  feature_names=feature_names,
                                                  categorical_features=feature_names,
                                                  class_names=['0', '1'],
                                                  discretize_continuous=True)

    # Function to evaluate model predictions
    def eval_model(my_model):
        def eval_closure(x):
            predictions = my_model.predict_proba(x)
            return predictions
        return eval_closure

    # Define a function that creates an evaluator for a given model.
    def evaluate_model(model):
        # Define a closure that takes input data and returns predictions using the provided model.
        def evaluate_closure(input_data):
            predictions = model.predict_proba(input_data)
            return predictions
        # Return the closure that encapsulates the evaluation function.
        return evaluate_closure

    # Create an evaluator for the best single model.
    best_model_evaluator = evaluate_model(tpot.fitted_pipeline_)

    return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer, best_model_evaluator

# Function to extract fragments from a set of SMILES strings
def extract_fragments(smiles_set):
    fragments = []
    for smiles in smiles_set:
        smiles = smiles.strip('{}')
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fragments.extend(Chem.GetMolFrags(mol, asMols=True))
    return fragments

# Function to generate circular fingerprint dictionary for a given molecule
def generate_circular_fingerprint_dict(molecule, fp_length=1024):
    fingerprint_dict = {}
    featurizer = dc.feat.CircularFingerprint(sparse=True, smiles=True, radius=4, size=fp_length)
    fingerprint_features = featurizer._featurize(molecule)
    for key, value in fingerprint_features.items():
        index = key % fp_length
        if index not in fingerprint_dict:
            fingerprint_dict[index] = set()
        fingerprint_dict[index].add(value['smiles'])
    return fingerprint_dict

# Function to predict from single SMILES input
def predict_from_single_smiles(single_smiles, featurizer_name):
    try:
        featurizer = Featurizer[featurizer_name]
        mol = Chem.MolFromSmiles(single_smiles)
        if mol is not None:
            features = featurizer.featurize([mol])[0]
            input_features = pd.DataFrame([features])

            # Extract fragments and generate circular fingerprints
            fragments_activations = generate_circular_fingerprint_dict(mol)

            # Display DataFrame of fragments
            st.subheader("Fragments from Single SMILES")
            if fragments_activations:
                fragments_data = []
                for index, smiles_set in fragments_activations.items():
                    for smiles in smiles_set:
                        mol_frag = Chem.MolFromSmiles(smiles)
                        if mol_frag:
                            fragments_data.append({
                                'Index': index,
                                'Fragment_SMILES': smiles,
                                'Fragment_Weight': features[index]
                            })
                df_fragments = pd.DataFrame(fragments_data)
                st.write(df_fragments)
            else:
                st.write("No fragments found.")

            return input_features
        else:
            st.warning(f"Invalid SMILES: {single_smiles}")
            return None
    except Exception as e:
        st.error(f"Error during single SMILES prediction: {e}")
        return None

# Function to predict from uploaded Excel file
def predict_from_uploaded_file(upload_file, featurizer_name):
    try:
        df = pd.read_excel(upload_file)
        st.subheader("Predictions from Uploaded Excel File")

        tpot_model, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer, best_model_evaluator = preprocess_and_model(
            df, 'SMILES', 'Activity', featurizer_name
        )

        # Display DataFrame of fragments
        st.subheader("Fragments from Uploaded Excel File")
        if 'IDs' in test_dataset.columns:
            fragments_data = extract_fragments(df['IDs'])
            df_fragments = pd.DataFrame(fragments_data)
            st.write(df_fragments)
        else:
            st.write("No fragments found.")

        return df
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")
        return None

# Main function for Streamlit app
def main():
    st.title('Molecule Activity Prediction')

    # Sidebar options
    st.sidebar.header('Select Featurizer')
    featurizer_name = st.sidebar.selectbox('Choose featurizer', list(Featurizer.keys()))

    st.sidebar.header('Modeling Parameters')
    generations = st.sidebar.slider('TPOT Generations', min_value=1, max_value=10, value=5, step=1)
    cv = st.sidebar.slider('Cross-validation folds', min_value=2, max_value=10, value=5, step=1)

    st.sidebar.header('Model Building')

    # File upload for model building
    st.sidebar.subheader('Upload Excel file for model building')
    model_build_file_key = "model_build_file"
    model_build_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'], key=model_build_file_key)

    if model_build_file is not None:
        try:
            df = pd.read_excel(model_build_file)
            st.subheader("Model Building")
            st.write("Building model from uploaded Excel file...")
            st.write("Please wait, this may take some time depending on the dataset size and complexity.")

            tpot_model, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer, best_model_evaluator = preprocess_and_model(
                df, 'SMILES', 'Activity', featurizer_name, generations, cv
            )

            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.write(f"Model Precision: {precision:.2f}")
            st.write(f"Model Recall: {recall:.2f}")
            st.write(f"Model F1 Score: {f1:.2f}")

            if roc_auc is not None:
                st.write(f"Model ROC AUC: {roc_auc:.2f}")

            st.subheader("Sample Predictions")
            st.write(df.head())

            st.sidebar.success("Model successfully built!")

        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    st.sidebar.header('Single Prediction')

    # Single SMILES prediction
    st.sidebar.subheader('Predict from Single SMILES')
    single_smiles_key = "single_smiles"
    single_smiles = st.sidebar.text_input("Enter SMILES", "", key=single_smiles_key)

    if st.sidebar.button("Predict Single SMILES"):
        if single_smiles:
            input_features = predict_from_single_smiles(single_smiles, featurizer_name)
            if input_features is not None:
                st.subheader("Input Features")
                st.write(input_features)

                # Interpret prediction using LIME
                if tpot_model is not None and X_train is not None:
                    st.subheader("Prediction Interpretation (LIME)")
                    explain = explainer.explain_instance(input_features.values[0], best_model_evaluator, num_features=10, top_labels=1)
                    st.write(explain.show_in_notebook())
        else:
            st.sidebar.warning("Please enter a valid SMILES.")

    st.sidebar.header('Batch Prediction')

    # Batch prediction from uploaded Excel file
    st.sidebar.subheader('Predict from Uploaded Excel file')
    upload_file_key = "upload_file"
    upload_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'], key=upload_file_key)

    if upload_file is not None:
        try:
            df_prediction = predict_from_uploaded_file(upload_file, featurizer_name)
            if df_prediction is not None:
                st.subheader("Batch Prediction Results")
                st.write(df_prediction)

                # Download predictions as CSV
                csv = df_prediction.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings
                linko= f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions as CSV</a>'
                st.markdown(linko, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during batch prediction: {e}")

if __name__ == '__main__':
    main()
