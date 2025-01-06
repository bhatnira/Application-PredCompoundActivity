# Description
This first Streamlit web application, created as part of this dissertation, allows users to predict the activity and potency of compounds as potential AChE inhibitors (see Figure 26). The application hosts the best models derived from each variant. Different input options are available, including the user’s preference to draw the molecule on the Canvas and produce predictions. All inputs are cleaned and standardized before predictions are made. Local LIME interpretation and atomic contribution mapping and visualization are also supported for GCN-based models and for descriptor/fingerprint-based models, respectively.

# Clone the Repo into VS Code
git clone https://github.com/bhatnira/ML-AcitivityPred-APP.git 
or, copy and paste the URL inside source control

# Create Local Environment and Activate
Open terminal, go to the repository directory, and create and activate local environment with the following command
python3 -m venv env && source ./env/bin/activate

# Install Application Dependencies within the Environment
pip install -r requirements.txt

# Run the Application
After installation of all the dependencies, you can run the program with the main application `app_choice.py`, which hosts all the other applications. 
streamlit run app_choice.py
