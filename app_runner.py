"""

"""
import streamlit as st
import importlib.util
import pkgutil
import apps


def list_apps():
    app_list = []
    for _, module_name, _ in pkgutil.iter_modules(apps.__path__):
        app_list.append(module_name)
    return app_list

st.sidebar.title("App Runner")
app_list = list_apps()
app_name = st.sidebar.selectbox("Select App", app_list)
app_module = importlib.import_module(f"apps.{app_name}")

