"""
This module is used to run different Streamlit applications. It provides a sidebar for the user to select the application they want to run.
"""

import streamlit as st
import importlib.util
import pkgutil
import apps

def list_apps():
    """
    This function lists all the modules in the 'apps' package.

    Returns:
        app_list (list): A list of module names in the 'apps' package.
    """
    app_list = []
    for _, module_name, _ in pkgutil.iter_modules(apps.__path__):
        app_list.append(module_name)
    return app_list

# Set the title of the sidebar
st.sidebar.title("App Runner")

# Get the list of applications
app_list = list_apps()

# Create a select box in the sidebar for the user to select the application they want to run
app_name = st.sidebar.selectbox("Select App", app_list)

# Import the selected application as a module
app_module = importlib.import_module(f"apps.{app_name}")
