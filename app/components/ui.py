from __future__ import annotations


import streamlit as st




def section(title: str):
    st.markdown(f"### {title}")




def subtle(text: str):
    st.caption(text)




def warn(text: str):
    st.warning(text, icon="⚠️")




def success(text: str):
    st.success(text)