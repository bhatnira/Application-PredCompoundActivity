Description
This Streamlit web application, created as part of this dissertation, allows users to predict the activity and potency of compounds as potential AChE inhibitors (see Figure 26). The application hosts the best models derived from each variant. It provides different input options, including the ability for users to draw a molecule on the Canvas and produce predictions. All inputs are cleaned and standardized before predictions are made. The application also supports local LIME interpretation and atomic contribution mapping and visualization. These features are available for GCN-based models and descriptor/fingerprint-based models, respectively.

Clone the Repository into VS Code
To clone the repository, use the following command:

bash
Copy code
git clone https://github.com/bhatnira/ML-AcitivityPred-APP.git
Alternatively, you can copy and paste the URL into your source control interface in VS Code.

Create and Activate a Local Environment
Open the terminal, navigate to the repository directory, and create and activate a local virtual environment using the following command:

bash
Copy code
python3 -m venv env && source ./env/bin/activate
Install Application Dependencies
Once the virtual environment is activated, install the necessary dependencies by running:

bash
Copy code
pip install -r requirements.txt
Run the Application
After installing all dependencies, you can start the application by running the main program app_choice.py, which hosts all other application modules. Use the following command:

bash
Copy code
streamlit run app_choice.py
