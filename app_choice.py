import streamlit as st
import subprocess
import os
import threading

# List of Streamlit apps
apps = {
    "ML Classification": "app_classification.py",
    "ML Regression": "app_regression.py",
    "GraphConv Classification Deployment": "graphD_C.py",
    "GraphConv Regression Deployment": "graphD_R.py",
    "GraphConV Classification Modeling": "graphM_C.py",
    "GraphConv Regression Modeling": "graphM_R.py"
}

# Function to start a Streamlit app
def start_app(app_path, port):
    command = ["streamlit", "run", app_path, "--server.port", str(port)]
    subprocess.run(command, check=True)

# Function to run an app in a separate thread
def run_app_in_thread(app_path, port):
    thread = threading.Thread(target=start_app, args=(app_path, port))
    thread.start()
    return thread

# Function to check if a port is available
def is_port_available(port):
    try:
        subprocess.check_output(["lsof", "-i", f":{port}"])
        return False
    except subprocess.CalledProcessError:
        return True

# Streamlit app layout
st.title("Streamlit App Selector")

# Dropdown menu for app selection
selected_app = st.selectbox("Select an app to open:", list(apps.keys()))

# Button to launch the selected app
if st.button("Run Selected App"):
    if selected_app:
        app_path = apps[selected_app]
        port = 8502 + list(apps.keys()).index(selected_app)
        
        # Check if the app is already running
        if not is_port_available(port):
            st.write(f"App '{selected_app}' is already running on port {port}.")
        else:
            st.write(f"Starting app '{selected_app}' on port {port}...")
            # Run the selected app in a new thread
            run_app_in_thread(app_path, port)
            st.write(f"App '{selected_app}' should now be running on port {port}.")
            st.write(f"Open the app at [http://localhost:{port}](http://localhost:{port})")
    else:
        st.write("Please select an app.")
