
import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import zipfile
from sklearn.metrics import accuracy_score, f1_score

# Helper functions
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                      os.path.join(path, '..')))

def plot_training_history(history):
    st.write("Training history plot placeholder.")

def plot_true_vs_pred(y_true, y_pred):
    st.write("True vs Predicted plot placeholder.")


def main():
    """Main function for the GraphConvNet Classification app."""
    st.title('🧬 GraphConvNet (Classification) for Compound Activity Prediction')

    # Sidebar controls
    st.sidebar.header("Configuration & Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file for training", type="xlsx", key="graphD_C_excel")
    show_data = st.sidebar.checkbox("Show Data Preview", value=True)
    show_metrics = st.sidebar.checkbox("Show Metrics", value=True)
    show_training = st.sidebar.checkbox("Show Training Section", value=True)
    st.sidebar.markdown("---")
    st.sidebar.header("App Options")
    user_name = st.sidebar.text_input("Your Name", "")
    if user_name:
        st.sidebar.write(f"👋 Hello, {user_name}!")

    # Main panel
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        columns = df.columns.tolist()
        st.header("Model Configuration")
        smiles_column = st.selectbox("Select the Smile column", columns)
        label_column = st.selectbox("Select the categorical label column", columns)
        batch_size = st.text_input("Batch size", "256")
        dropout = st.text_input("Dropout rate", "0.1")
        nb_epoch = st.text_input("Number of epochs", "120")
        graph_conv_layers = st.text_input("Graph convolution layers (comma-separated)", "64,64")
        test_size = st.text_input("Test size (fraction)", "0.15")
        valid_size = st.text_input("Validation size (fraction)", "0.15")

        st.markdown("---")

        # Data Preview
        if show_data:
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(20))

        st.markdown("---")

        # Metrics Display
        if show_metrics:
            st.subheader("📈 Model KPIs (Demo)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Samples", f"{len(df)}", "+5%")
            col2.metric("Features", f"{len(df.columns)}", "+2")
            col3.metric("Ready for Training", "Yes" if smiles_column and label_column else "No", "")

        st.markdown("---")

        # Training Section
        if show_training:
            with st.expander("🚀 Train & Download Model", expanded=True):
                if st.button("Train Model", type="primary"):
                    st.write("Training the model...")
                    model_dir = "./trained_model"
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir)
                    os.makedirs(model_dir)

                    batch_size_val = int(batch_size)
                    dropout_val = float(dropout)
                    nb_epoch_val = int(nb_epoch)
                    test_size_val = float(test_size)
                    valid_size_val = float(valid_size)
                    graph_conv_layers_val = [int(layer) for layer in graph_conv_layers.split(',')]

                    # Dummy train_model stub (replace with actual logic)
                    model, test_dataset, training_history = None, None, None
                    try:
                        # TODO: Replace with actual training logic
                        st.info("Model training logic goes here.")
                        # model, test_dataset, training_history = train_model(...)
                    except Exception as e:
                        st.error(f"Training failed: {e}")

                    if model is not None:
                        st.success("Model training completed.")
                        if test_dataset is not None:
                            y_true = np.array(test_dataset.y).ravel()
                            y_pred = model.predict(test_dataset).ravel()
                            accuracy = accuracy_score(y_true, y_pred)
                            f1 = f1_score(y_true, y_pred, average='weighted')
                            st.write("Test Accuracy:", accuracy)
                            st.write("Test F1 Score:", f1)
                            st.write("Training History:")
                            plot_training_history(training_history)
                            st.write("True vs Predicted Values:")
                            plot_true_vs_pred(y_true, y_pred)
                            with zipfile.ZipFile('trained_model.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                                zipdir(model_dir, zipf)
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
        st.info("Please upload an Excel file in the sidebar to begin.")

    # App description in expander
    with st.expander("ℹ️ About this App", expanded=False):
        st.write("""
        This app demonstrates a modern UI for compound activity prediction using a Graph Convolutional Network (GCN).
        - Use the sidebar to upload your data and configure model parameters.
        - Preview your data, view key metrics, and train/download your model.
        - UI is responsive and uncluttered, with expandable sections for details.
        """)

if __name__ == "__main__":
    main()
