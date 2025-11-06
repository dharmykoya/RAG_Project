import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader, JSONLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from pathlib import Path
from typing import List, Any


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load documents from various file types in the specified directory.

    Args:
        data_dir (str): The directory containing the documents.                     
    Returns:
        List[Any]: A list of loaded documents.
    """
    loaders = [
        (".pdf", PyPDFLoader),
        (".pdf", PyMuPDFLoader),
        (".docx", Docx2txtLoader),
        (".txt", TextLoader),
        (".json", JSONLoader),
        (".csv", CSVLoader),
        (".xlsx", UnstructuredExcelLoader),
    ]

    all_documents = []
    for ext, loader_class in loaders:
        try:
            loader = DirectoryLoader(data_dir, glob=f"**/*{ext}", loader_cls=loader_class)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {ext} files: {e}")

    return all_documents