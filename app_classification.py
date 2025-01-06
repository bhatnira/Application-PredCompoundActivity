import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
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

#st.set_option('deprecation.showPyplotGlobalUse', False)

ssl._create_default_https_context = ssl._create_unverified_context

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

# Function to preprocess data and perform modeling for classification
def preprocess_and_model(df, smiles_col, activity_col, featurizer_name, generations=5, cv=5, verbosity=2, test_size=0.20):
    # Standardize Smile
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
            st.warning(f"Invalid Smile: {smiles}")

    if not features:
        st.error("No valid molecules found for featurization. Please ensure your Smile data is correct.")
        return None, None, None, None, None, None, None, None, None, None

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
    #st.write(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    st.subheader("Confusion Matrix Heatmap")
    st.pyplot(fig)

    # Print best pipeline
    st.subheader("Best TPOT Pipeline")
    st.write(tpot.fitted_pipeline_)

    # Save TPOT model and X_train separately
    model_filename = 'best_model.pkl'
    X_train_filename = 'X_train.pkl'

    with open(model_filename, 'wb') as f_model:
        joblib.dump(tpot.fitted_pipeline_, f_model)
    
    with open(X_train_filename, 'wb') as f_X_train:
        joblib.dump(X_train, f_X_train)

    # Create download links for the model files
    st.markdown(create_downloadable_model_link(model_filename, 'Download Best Model'), unsafe_allow_html=True)
    st.markdown(create_downloadable_model_link(X_train_filename, 'Download X_train'), unsafe_allow_html=True)

    # Get feature names used in modeling
    feature_names = list(X_train.columns)

    return tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer

# Function to create a downloadable link for HTML content
def create_download_link(html_content, link_text):
    href = f'<a href="data:text/html;base64,{base64.b64encode(html_content.encode()).decode()}" download="{link_text}.html">{link_text}</a>'
    return href

# Function to create a downloadable link for model files
def create_downloadable_model_link(model_filename, link_text):
    with open(model_filename, 'rb') as f:
        model_data = f.read()
    b64 = base64.b64encode(model_data).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="{model_filename}">{link_text}</a>'
    return href

# Function to interpret prediction using LIME
def interpret_prediction(tpot_model, input_features, X_train):
    # Create LIME explainer using X_train
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="classification",
        feature_names=X_train.columns,
        class_names=["Not Active", "Active"],
        verbose=True,
        discretize_continuous=True
    )
    
    explanation = explainer.explain_instance(
        input_features.values[0],
        tpot_model.predict_proba,
        num_features=len(input_features.columns)
    )

    # Generate HTML explanation
    html_explanation = explanation.as_html()
    return html_explanation

# Function to predict from single Smile input
def predict_from_single_smiles(single_smiles, featurizer_name='Circular Fingerprint'):
    standardized_smiles = standardize_smiles(single_smiles)
    mol = Chem.MolFromSmiles(standardized_smiles)
    
    if mol is not None:
        featurizer = Featurizer[featurizer_name]
        features = featurizer.featurize([mol])[0]
        feature_df = pd.DataFrame([features], columns=[f"fp_{i}" for i in range(len(features))])
        feature_df = feature_df.astype(float)

        # Load trained model and X_train
        try:
            with open('best_model.pkl', 'rb') as f_model, open('X_train.pkl', 'rb') as f_X_train:
                tpot_model = joblib.load(f_model)
                X_train = joblib.load(f_X_train)
        except FileNotFoundError:
            st.warning("Please build and save the model in the 'Build Model' section first.")
            return None, None

        # Predict using the trained model
        prediction = tpot_model.predict(feature_df)[0]
        probability = tpot_model.predict_proba(feature_df)[0][1] if hasattr(tpot_model, 'predict_proba') else None

        # Interpret prediction using LIME
        explanation_html = interpret_prediction(tpot_model, feature_df, X_train)

        return prediction, probability, explanation_html
    else:
        st.warning("Invalid Smile input. Please check your input and try again.")
        return None, None, None

# Main Streamlit application
def main():
    # Initialize selected featurizer name session variable
    if 'selected_featurizer_name' not in st.session_state:
        st.session_state.selected_featurizer_name = list(Featurizer.keys())[0]  # Set default featurizer

    st.title("Chemical Activity Prediction(Classification)")

    # Navigation
    options = ["Home", "Build Model", "Predict from Smile", "Predict from Uploaded Excel File"]
    choice = st.sidebar.selectbox("Select Option", options)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Chemical Activity Prediction App!")
        st.write("Select an option from the sidebar to proceed.")

    elif choice == "Build Model":
        st.subheader("Build Model")
        st.write("Upload an Excel file containing Smile and corresponding activity labels to train the model.")
        uploaded_file = st.file_uploader("Upload Excel file with Smile and Activity", type=["xlsx"])

        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Show dataframe
            st.subheader("Uploaded Data for Model Building")
            st.write(df)

            # Select Smile and Activity columns
            col_names = df.columns.tolist()
            smiles_col = st.selectbox("Select Smile Column", col_names, key='smiles_column')
            activity_col = st.selectbox("Select Activity Column", col_names, key='activity_column')

            # Select featurizer
            st.session_state.selected_featurizer_name = st.selectbox("Select Featurizer", list(Featurizer.keys()), key='featurizer_name', index=list(Featurizer.keys()).index(st.session_state.selected_featurizer_name))

            # Model training parameters
            generations = st.sidebar.slider("Number of Generations", min_value=1, max_value=50, value=5)
            cv = st.sidebar.slider("Number of Cross-Validation Folds", min_value=1, max_value=50, value=5)
            verbosity = st.sidebar.slider("Verbosity", min_value=0, max_value=2, value=2)
            test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.05)

            # Build and train the model
            if st.button("Build and Train Model"):
                st.write("Building and training the model...")
                tpot, accuracy, precision, recall, f1, roc_auc, X_test, y_test, y_pred, df, X_train, y_train, featurizer = preprocess_and_model(
                    df, smiles_col, activity_col, st.session_state.selected_featurizer_name, generations=generations, cv=cv, verbosity=verbosity, test_size=test_size)

                # Display model metrics
                st.subheader("Model Evaluation Metrics")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")
                if roc_auc is not None:
                    st.write(f"ROC AUC: {roc_auc:.2f}")

    elif choice == "Predict from Smile":
        st.subheader("Predict from Smile")
        smile_input = st.text_input("Enter Smile for prediction")

        if st.button("Predict from Smile"):
            if smile_input:
                prediction, probability, explanation_html = predict_from_single_smiles(smile_input, st.session_state.selected_featurizer_name)
                if prediction is not None:
                    st.write(f"Predicted Activity: {'Active' if prediction == 1 else 'Not Active'}")
                    if probability is not None:
                        st.write(f"Probability: {probability:.4f}")

                    # Display LIME explanation as HTML
                    st.markdown(create_download_link(explanation_html, "Explanation"), unsafe_allow_html=True)
                else:
                    st.warning("Failed to make prediction. Please check your input and try again.")

    elif choice == "Predict from Uploaded Excel File":
        st.subheader("Predict from Uploaded Excel File")
        uploaded_file = st.file_uploader("Upload Excel file with Smile for prediction", type=["xlsx"])

        if uploaded_file is not None:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Show dataframe
            st.subheader("Uploaded Data for Prediction")
            st.write(df)

            # Select Smile column for prediction
            col_names = df.columns.tolist()
            smiles_col_predict = st.selectbox("Select Smile Column for Prediction", col_names, key='smiles_column_predict')

            # Load trained model and X_train
            try:
                with open('best_model.pkl', 'rb') as f_model, open('X_train.pkl', 'rb') as f_X_train:
                    tpot_model = joblib.load(f_model)
                    X_train = joblib.load(f_X_train)
            except FileNotFoundError:
                st.warning("Please build and save the model in the 'Build Model' section first.")
                return

            # Button to trigger prediction
            if st.button("Predict from Uploaded File"):
                if smiles_col_predict in df.columns:
                    predictions = []
                    probabilities = []
                    explanations = []

                    for index, row in df.iterrows():
                        standardized_smiles = standardize_smiles(row[smiles_col_predict])
                        if standardized_smiles:
                            mol = Chem.MolFromSmiles(standardized_smiles)
                            if mol is not None:
                                featurizer = Featurizer[st.session_state.selected_featurizer_name]
                                features = featurizer.featurize([mol])[0]
                                feature_df = pd.DataFrame([features], columns=[f"fp_{i}" for i in range(len(features))])
                                feature_df = feature_df.astype(float)

                                # Predict using the trained model
                                prediction = tpot_model.predict(feature_df)[0]
                                probability = tpot_model.predict_proba(feature_df)[0][1] if hasattr(tpot_model, 'predict_proba') else None

                                # Interpret prediction using LIME
                                explanation_html = interpret_prediction(tpot_model, feature_df, X_train)

                                # Store results
                                predictions.append(prediction)
                                probabilities.append(probability)
                                explanations.append(explanation_html)

                                # Display chemical structure with Smile on the left
                                st.subheader(f"Prediction {index + 1}")
                                st.image(Chem.Draw.MolToImage(mol, size=(300, 300), kekulize=True), caption=row[smiles_col_predict], use_column_width=True)

                                # Display prediction and probability
                                st.write(f"Predicted Activity: {'Active' if prediction == 1 else 'Not Active'}")
                                if probability is not None:
                                    st.write(f"Probability: {probability:.4f}")

                                # Display LIME explanation as HTML on the right
                                explanation_link = create_download_link(explanation_html, f"Explanation {index + 1}")
                                st.markdown(explanation_link, unsafe_allow_html=True)

                            else:
                                st.warning(f"Invalid Smile input at row {index + 1}. Skipping prediction.")

                        else:
                            st.warning(f"Invalid Smile input at row {index + 1}. Skipping prediction.")

                    # Add predictions, probabilities, and explanations to DataFrame
                    df['Prediction'] = predictions
                    df['Probability'] = probabilities
                    df['Explanation'] = explanations

                    # Display updated DataFrame with predictions
                    st.subheader("Predictions with Explanations")
                    st.write(df)

                else:
                    st.warning("Smile column not found in the uploaded file. Please select the correct column.")

if __name__ == "__main__":
    main()
