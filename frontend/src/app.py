# import os
import streamlit as st

st.set_page_config(page_title="Streamlit UI", page_icon="📊")

st.title("Streamlit UI")
st.write("Simple app")

name = st.text_input("What is your name?", "")
if name:
    st.success(f"Hello, {name}!")
