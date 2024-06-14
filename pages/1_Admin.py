#!/usr/bin/env python

# Ragai - (c) Eric Dodémont, 2024.

"""
This subpage runs the admin web interface.
"""

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
import os
import zipfile
import subprocess

from modules.web_scraping_utils_v1 import scrape_commons_category, scrape_web_page_url
from modules.utils_v1 import load_files_and_embed, delete_directory
from config.config import *


def reset_conversation():
    """
    Reset the conversation: clear the chat history and clear the screen.
    """

    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.chat_history2 = ConversationBufferWindowMemory(k=4, return_messages=True)


def unzip_and_replace(file_path):
    # Check if the file is a zip file
    if zipfile.is_zipfile(file_path):
        # Create a ZipFile object
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Extract all the contents of the zip file in the same directory
            zip_ref.extractall(os.path.dirname(file_path))
        
        # Remove the original zip file
        os.remove(file_path)
        print(f"The file {file_path} has been unzipped and the original zip file has been removed.")
    else:
        print(f"The file {file_path} is not a zip file.")


st.set_page_config(page_title=ASSISTANT_NAME, page_icon=ASSISTANT_ICON)

if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE

if "password_ok" not in st.session_state:
    st.session_state.password_ok = False

if "input_password" not in st.session_state:
    st.session_state.input_password = ""

st.title("Admin")

st.sidebar.write(f"Model: {st.session_state.model} ({st.session_state.temperature})")

# Ask admin password to access admin menu
admin_password = os.getenv("ADMIN_PASSWORD", "YYYY")
input_password = st.sidebar.text_input("Enter admin password: ", type="password", value=st.session_state.input_password)
st.session_state.input_password = input_password
if st.session_state.input_password != admin_password:
    st.session_state.password_ok = False
else:
    st.session_state.password_ok = True

if st.session_state.password_ok:

    # # # # # # # # # # # # # # # # # # # # #
    # Side bar window: second page (Admin)  #
    # # # # # # # # # # # # # # # # # # # # #
    
    options = ['Upload PDF Files', 'Upload JSON Files (Web Pages)', 'Upload JSON Files (Web Pages) in ZIP Format', 'Scrape Web Pages', 'Scrape Web Pages from Wikimedia Commons', 'Embed Pages in DB', 'Model and Temperature', 'Upload File']
    choice = st.sidebar.radio("Make your choice: ", options)

    if choice == "Scrape Web Pages":
        st.caption("Give the web page URL and the filter (CSS class). The page will be scraped and saved in a JSON file in the 'json_files' directory.")
        st.caption("""
                    Filter: 
                    - two-third last (balat / irpa)
                    - media-body (belgica / kbr)
                    - hproduct commons-file-information-table (commons / wikimedia: summary or description section)
                    - card metadata-box-card mb-3 (europeana / kul, irpa, etc.)
                    """)
        url = st.text_input("URL: ")
        filter = st.text_input("Filter: ")
        if url and filter:
            st.write(f"Scraping the web page...")
            scrape_web_page_url(url, filter)
            st.write(f"Web page scraped and saved in a JSON file!")

    elif choice == "Model and Temperature":
        st.caption("Change the model and the temperature for the present chat session.")
        model_list = [OPENAI_MENU, ANTHROPIC_MENU, VERTEXAI_MENU, OLLAMA_MENU]
        st.session_state.model = st.selectbox('Model: ', model_list, DEFAULT_MENU_CHOICE)
        st.session_state.temperature = st.slider("Temperature: ", 0.0, 2.0, DEFAULT_TEMPERATURE)
        st.caption("OpenAI: 0-2, Anthropic: 0-1")

    elif choice == "Scrape Web Pages from Wikimedia Commons":
        st.caption("Give a category name from Wikimedia Commons. The pages will be scraped and saved in a JSON file in the 'json_files' directory.")
        category = st.text_input("Category: ")
        if category:
            st.write(f"Scraping the web pages...")
            scrape_commons_category(category)
            st.write(f"Web pages scraped and saved in a JSON file!")

    elif choice == "Upload File":
        st.caption("Upload a file in the 'root' directory.")
        uploaded_file = st.file_uploader("Choose a file:")
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            file_name = uploaded_file.name
            with open(file_name, "wb") as file:
                file.write(bytes_data)
            st.success(f"File '{file_name}' uploaded and saved successfully!")
        else:
            st.warning("No file uploaded yet.")

    elif choice == "Upload PDF Files":
        st.caption("Upload PDF files in the 'pdf_files' directory.")
        uploaded_files = st.file_uploader("Choose PDF files:", type=["pdf"], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                file_name = uploaded_file.name
                with open(f"./pdf_files/{file_name}", "wb") as file:
                    file.write(bytes_data)
                st.success(f"File '{file_name}' uploaded and saved successfully!")
            else:
                st.warning("No file uploaded yet.")  

    elif choice == "Upload JSON Files (Web Pages)":
        st.caption("Upload JSON files (Web Pages) in the 'json_files' directory. One or many JSON items (Web pages) per JSON file.")
        uploaded_files = st.file_uploader("Choose JSON files:", type=["json"], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                file_name = uploaded_file.name
                with open(f"./json_files/{file_name}", "wb") as file:
                    file.write(bytes_data)
                st.success(f"File '{file_name}' uploaded and saved successfully!")
            else:
                st.warning("No file uploaded yet.")

    elif choice == "Upload JSON Files (Web Pages) in ZIP Format":
        st.caption("Upload JSON files (Web Pages) in the 'json_files' directory. One or many JSON items (Web pages) per JSON file. The ZIP files will be unziped.")
        uploaded_files = st.file_uploader("Choose ZIP files:", type=["zip"], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                file_name = uploaded_file.name
                with open(f"./json_files/{file_name}", "wb") as file:
                    file.write(bytes_data)
                unzip_and_replace(f"./json_files/{file_name}")
                st.success(f"File '{file_name}' uploaded and unziped successfully!")
            else:
                st.warning("No file uploaded yet.")  

    elif choice == "Embed Pages in DB":
        # Embed data in Chroma DB
        # Load and index

        st.caption('Embed all the web and pdf pages in the Chroma vector DB.')
        st.caption('Caution: Works only with the Chroma vector DB located on the same server as the app!')

        JSON_FILES_DIR = "./json_files/"
        PDF_FILES_DIR = "./pdf_files/"

        # JSON files
        json_files = os.listdir(JSON_FILES_DIR)
        json_paths = []
        for json_file in json_files:
            json_path = f"{JSON_FILES_DIR}{json_file}"
            json_paths.append(json_path)

        # PDF files
        pdf_files = os.listdir(PDF_FILES_DIR)
        pdf_paths = []
        for pdf_file in pdf_files:
            pdf_path = f"{PDF_FILES_DIR}{pdf_file}"
            pdf_paths.append(pdf_path)

        if st.button("Start Embed (locally only)"):
            load_files_and_embed(json_paths, pdf_paths, embed=True)
            st.write("Done!")

        if st.button("Delete DB (locally only)"):
            delete_directory("./chromadb")
            st.write("Done!")

        if st.button("Restart DB (locally only)"):
            command = ['bash', './db.sh', 'restart']
            result = subprocess.run(command, capture_output=True, text=True)
            st.write(result.stdout)
            st.write(result.stderr)
            st.write("Done!")

        if st.button("Clear Memory and Streamlit Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            reset_conversation()
            st.write("Done!")

        if st.button("Files and DB Info (locally only)"):

            load_files_and_embed(json_paths, pdf_paths, embed=False)

            st.write(f"The Chroma vector DB is located on {CHROMA_SERVER_HOST}:{CHROMA_SERVER_PORT}.")

            try:

                file_path = './chromadb/chroma.sqlite3'
                file_size = os.path.getsize(file_path)
                file_size = file_size / 1024  # In KB
                if file_size > 144:
                    st.write(f"DB size: {file_size} KB")
                else:
                    st.write(f"DB size: {file_size} KB. DB is empty!")

                path = './chromadb'
                files = os.listdir(path)
                st.write("DB path:")
                st.write(files)

            except Exception as e:
                st.write("The Chroma vector DB is not available locally.")
                st.write(f"Error: {e}")

            try:

                path = './'
                files = os.listdir(path)
                st.write("Root path:")
                st.write(files)

            except Exception as e:
                st.write(f"Error: {e}")
