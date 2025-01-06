

# Streamlit Application for Development and Deployment of AutoML TPOT and GCN Models for Compound Activity Prediction

A second Streamlit web application was created to serve as a tool for the development and simultaneous deployment of two advanced machine learning models: the **feature/descriptor-based AutoML TPOT** and the **Graph Convolutional Network (GCN)**. These models are designed for **compound activity prediction** (classification) and **potency prediction** (regression), provided an adequate amount of quality data is available. The application also includes model interpretation tools, such as:
- **Atomic contribution maps** for predictions made by the **GCN**.
- **Feature importance explanations** using **LIME interpretation** for the AutoML TPOT model’s predictions.

The main application hosts **six modules**, each with a specific purpose:
1. **AutoML TPOT-based compound activity prediction (classification)** model development and deployment.
2. **AutoML TPOT-based compound potency prediction (regression)** model development and deployment.
3. **GCN-based model development** for activity prediction (classification).
4. **GCN-based model deployment** for activity prediction (classification).
5. **GCN-based model development** for potency prediction (regression).
6. **GCN-based model deployment** for potency prediction (regression).

### Key Features:
- **Model Development**:  
  For model development, users can upload an Excel file with SMILES and corresponding activity/potency data. The application will automatically build models based on the provided parameters and evaluate them using various metrics and visualizations. Input SMILES are cleaned and standardized using **RDKit modules**, as described in the methods section.
  
- **Model Featurization**:  
  For AutoML TPOT modeling, the user can choose from one of six featurization methods:
  1. **RDKit Descriptors**
  2. **Extended Connectivity Circular Fingerprint (ECFP)**
  3. **PubChem Fingerprint**
  4. **MACCS Keys**
  5. **Mordred**
  6. **Mol2Vec**
  
  The GCN modeling process is fixed to use the **ConvMolFeaturizer** from **DeepChem**.

- **Model Evaluation**:  
  Evaluation metrics for **classification models** (both AutoML TPOT and GCN) include:
  - **ROC curve** and **AUC-ROC score**
  - **Accuracy**, **Precision**, **Recall**, **F1 score**
  - **Confusion matrix**
  
  For **regression models**, evaluation metrics include:
  - **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **R-squared**
  - **True vs. predicted scatter plot**
  - **Best pipeline** (for AutoML TPOT models)

- **Deployment**:  
  The deployment module allows separate predictions for activity and potency. It also includes:
  - **LIME interpretation** for **AutoML-based model predictions** (classification or regression).
  - **Atomic contribution map visualizations** for the **GCN model's predictions** (both regression and classification).

  Note that GCN models (whether for regression or classification) must be uploaded separately in the deployment module to maintain the integrity of the models.

This application is designed to assist researchers in developing and deploying machine learning models for compound activity and potency prediction, with powerful interpretability tools to understand the model's decisions.

1. **Clone the Repository**  
Clone the repository to your local machine using the following command:

    ```bash
    git clone https://github.com/bhatnira/ML-AcitivityPred-APP.git
    ```

    Alternatively, you can copy and paste the URL into your IDE's source control.

2. **Set Up a Virtual Environment**  
Navigate to the cloned repository folder and create a virtual environment to manage dependencies:

    ```bash
    python3 -m venv env
    ```

    Activate the environment:

    - **On macOS/Linux:**
    
      ```bash
      source ./env/bin/activate
      ```

    - **On Windows:**
    
      ```bash
      .\env\Scripts\activate
      ```

3. **Install Dependencies**  
Once the virtual environment is activated, install the necessary dependencies listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**  
After the dependencies are installed, launch the application by running the following command:

    ```bash
    streamlit run app_choice.py
    ```

The app will open in your default browser, and you can start using it to predict AChE inhibitor activity and potency.

The application supports both **Graph Convolutional Network (GCN)-based models** and **descriptor/fingerprint-based models**. LIME-based explanations provide insights into model predictions, helping users understand which features contributed the most to the predicted activity. This tool is designed for researchers and practitioners working on drug discovery and computational chemistry.

Feel free to explore the code, make modifications, and contribute to improving this tool.

For any questions or issues, please open an [issue](https://github.com/bhatnira/ML-AcitivityPred-APP/issues) in the repository.
