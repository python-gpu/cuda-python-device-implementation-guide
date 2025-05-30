# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

project = 'CUDA Python Device Implementation Guide'
copyright = '2025, NVIDIA Corporation'
author = 'NVIDIA Corporation'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # for google style support
    'myst_parser'  # for markdown support
]

master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = []

# -- autodoc configuration ----------------------------------------------------
autodoc_typehints = "none"
add_module_names = False
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'nvidia_sphinx_theme'
html_show_sphinx = False

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

html_baseurl = "https://andy.terrel.us/cuda-python-device-implementation-guide/"

html_static_path = ['cuda-python-device-implementation-guide/_static']
html_css_files = [
            'cuda-python-device-implementation-guide/_static/custom.css',
            ]
