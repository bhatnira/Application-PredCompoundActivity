"""Main entry point for the Co# Define # Define the pages with their main functions
app_pages = {
    "Dashboard": None,
    "TPOT Classification": classification_main,
    "TPOT Regression": regression_main,
    "GraphConv Deploy (Class)": graphD_C_main,
    "GraphConv Deploy (Reg)": graphD_R_main,
    "GraphConv Train (Class)": graphM_C_main,
    "GraphConv Train (Reg)": graphM_R_main
}with their main functions
app_pages = {
    "Dashboard": None,
    "TPOT Classification": classification_main,
    "TPOT Regression": regression_main,
    "GraphConv Deploy (Class)": graphD_C_main,
    "GraphConv Deploy (Reg)": graphD_R_main,
    "GraphConv Train (Class)": graphM_C_main,
    "GraphConv Train (Reg)": graphM_R_main
}ivity Prediction Suite."""
import streamlit as st
from PIL import Image

# Import each app's main function
from app_classification import main as classification_main
from app_regression import main as regression_main
from graphD_C import main as graphD_C_main
from graphD_R import main as graphD_R_main
from graphM_C import main as graphM_C_main
from graphM_R import main as graphM_R_main

# Set page config for the entire app
st.set_page_config(
    page_title="Compound Activity Prediction Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header with logo and title
header_container = st.container()
with header_container:
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/DNA_Icon.png/240px-DNA_Icon.png", width=80)
    with col2:
        st.title("Compound Activity Prediction Suite")
    st.markdown("---")

# Define the pages with their icons and main functions
app_pages = {
    "🏠 Dashboard": None,
    "🔬 TPOT Classification": classification_main,
    "� TPOT Regression": regression_main,
    "⚛️ GraphConv Deploy (Class)": graphD_C_main,
    "� GraphConv Deploy (Reg)": graphD_R_main,
    "🧪 GraphConv Train (Class)": graphM_C_main,
    "🎯 GraphConv Train (Reg)": graphM_R_main
}

# Create horizontal navigation tabs
page = st.tabs(list(app_pages.keys()))

st.markdown("""
<style>
/* Global Styles */
.main {
    background-color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Hide Streamlit elements */
[data-testid="stHeader"] {display: none;}
div[data-testid="stToolbar"] {display: none;}
section[data-testid="stSidebar"] {display: none;}

/* Button Styles */
.stButton>button {
    color: #444444;
    background: #ffffff;
    border: 1px solid #dddddd;
    border-radius: 4px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    border-color: #666666;
    color: #000000;
}

/* Navigation Tabs */
div.stTabs {
    margin: 1rem 0 3rem 0;
}

/* Tab Container */
div[data-testid="stHorizontalBlock"] {
    gap: 10px !important;
    padding: 0.5rem;
    background: #f8fafc;
    border-radius: 8px;
}

/* Individual Tabs */
div.stTabs button {
    min-width: 200px;
    max-width: none;
    font-size: 14px;
    font-weight: 600;
    background: #ffffff;
    text-transform: none;
    letter-spacing: normal;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin: 0 5px;
    white-space: normal;
    height: auto !important;
    padding: 12px 20px;
}

div.stTabs button:hover {
    background: #edf2f7;
    color: #2c5282;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.08);
}

div.stTabs button[data-baseweb="tab"] {
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    min-height: 52px;
    height: auto;
    padding: 12px 20px;
    word-wrap: break-word;
}

div.stTabs button[aria-selected="true"] {
    background: #2c5282 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(44,82,130,0.2);
}

/* Tab content spacing */
div[data-testid="stVerticalBlock"] {
    gap: 2rem !important;
    padding: 1rem;
}

/* Active tab indicator */
div.stTabs button[aria-selected="true"]::after {
    content: "";
    position: absolute;
    bottom: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 6px solid #2c5282;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #1a202c;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* Markdown text */
.element-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #2d3748;
    line-height: 1.6;
}

/* File Uploader Styles */
[data-testid="stFileUploader"] {
    width: 100%;
    max-width: 100%;
}

/* Label style */
[data-testid="stFileUploader"] label {
    color: #2d3748 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
    display: block !important;
    letter-spacing: -0.01em !important;
}

[data-testid="stFileUploadDropzone"] {
    width: 100% !important;
    max-width: 100% !important;
    min-height: auto !important;
    padding: 16px !important;
    background: #ffffff !important;
    border: none !important;
    margin-top: 4px !important;
}

[data-testid="stFileUploadDropzone"] > div {
    width: 100% !important;
    max-width: 100% !important;
}

/* Main content area */
[data-testid="stFileUploadDropzone"] > div > div:first-child {
    display: none !important;
}

/* Upload text container */
[data-testid="stFileUploadDropzone"] div[data-testid="stMarkdownContainer"] {
    display: none !important;
}

/* File size limit and type */
[data-testid="stFileUploadDropzone"] small {
    font-size: 12px !important;
    color: #64748b !important;
    display: block !important;
    text-align: center !important;
    margin: 4px 0 0 0 !important;
}

/* Button container */
[data-testid="stFileUploadDropzone"] > div > div:last-child {
    padding: 0 !important;
    width: 100% !important;
}

/* Browse files button */
[data-testid="stFileUploadDropzone"] button {
    width: 100% !important;
    min-height: 42px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #2d3748 !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    display: block !important;
    letter-spacing: 0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

[data-testid="stFileUploadDropzone"] button:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e0 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

/* Column Styles */
[data-testid="column"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# Display content based on selected tab
with page[0]:  # Home
    st.markdown("""
    Welcome to the Compound Activity Prediction Suite! 🧬
    
    **Features:**
    - Build and train models for chemical activity prediction (classification & regression)
    - Use advanced Graph Convolutional Networks for molecular data
    - Predict from SMILES or batch Excel files
    - Download trained models for deployment
    - Visualize results and interpret predictions
    """)
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", 
             caption="AI for Chemistry", use_column_width=True)

# Handle other tabs
for i, (page_name, main_func) in enumerate(app_pages.items()):
    if i > 0:  # Skip Home tab
        with page[i]:
            # Create columns for controls and main content
            control_col, content_col = st.columns([1, 3])
            
            with control_col:
                st.markdown("### Configuration")
                
                # Common controls
                uploaded_file = st.file_uploader("Upload Excel file", type="xlsx", key=f"excel_{i}")
                
                # Model controls section
                st.markdown("#### Model Settings")
                config = {
                    "uploaded_file": uploaded_file,
                    "show_data": True,
                    "show_metrics": True,
                    "show_training": True
                }
                
                if "Classification" in page_name:
                    config["threshold"] = st.slider("Classification Threshold", 0.0, 1.0, 0.5, key=f"threshold_{i}")
                    config["class_weights"] = st.checkbox("Use Class Weights", key=f"weights_{i}")
                
                if "GraphConv" in page_name:
                    config["batch_size"] = st.number_input("Batch Size", 32, 512, 256, step=32, key=f"batch_{i}")
                    config["epochs"] = st.number_input("Epochs", 10, 200, 100, step=10, key=f"epochs_{i}")
                    config["dropout"] = st.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.1, key=f"dropout_{i}")
                    config["graph_conv_layers"] = st.text_input("Graph Conv Layers", "64,64", key=f"layers_{i}")
                    config["learning_rate"] = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f", key=f"lr_{i}")
                else:
                    # TPOT specific controls
                    if not "GraphConv" in page_name:
                        config["featurizer"] = st.selectbox(
                            "Featurization Method",
                            ["Circular Fingerprint", "MACCSKeys", "Mordred", "RDKit", "PubChem", "Mol2Vec"],
                            key=f"featurizer_{i}"
                        )
                        config["generations"] = st.number_input("TPOT Generations", 10, 100, 20, key=f"gen_{i}")
                        config["population"] = st.number_input("Population Size", 10, 100, 20, key=f"pop_{i}")
                        config["cv"] = st.number_input("Cross Validation Folds", 3, 10, 5, key=f"cv_{i}")
                
                # Data splitting
                st.markdown("#### Data Splitting")
                config["test_size"] = st.slider("Test Set Size", 0.1, 0.3, 0.2, 0.05, key=f"test_{i}")
                if "GraphConv" in page_name:
                    config["valid_size"] = st.slider("Validation Set Size", 0.1, 0.3, 0.2, 0.05, key=f"valid_{i}")
                
                # Visualization controls
                st.markdown("#### Display Options")
                config["show_data"] = st.checkbox("Show Data Preview", value=True, key=f"show_data_{i}")
                config["show_metrics"] = st.checkbox("Show Metrics", value=True, key=f"show_metrics_{i}")
                if "GraphConv" in page_name:
                    config["show_training"] = st.checkbox("Show Training Progress", value=True, key=f"show_train_{i}")
                config["show_plots"] = st.checkbox("Show Plots", value=True, key=f"show_plots_{i}")
                
            with content_col:
                if main_func and uploaded_file is not None:
                    main_func(config)
                elif uploaded_file is None:
                    st.info("Please upload an Excel file to begin.")
