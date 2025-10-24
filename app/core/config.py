# app/core/config.py
import os
import streamlit as st

def get_odds_api_key() -> str:
    if "ODDS_API_KEY" in st.secrets:
        return st.secrets["ODDS_API_KEY"]
    if "api_keys" in st.secrets and "the_odds_api" in st.secrets["api_keys"]:
        return st.secrets["api_keys"]["the_odds_api"]
    if os.getenv("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    raise RuntimeError("No API key found. Add it to Streamlit secrets or as an environment variable.")

