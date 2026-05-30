import streamlit as st
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

st.set_page_config(
    page_title='AgroPredict Colombia',
    page_icon='\U0001F33E',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .main-header h1 { color: #2E7D32; font-size: 2.5rem; margin-bottom: 0; }
    .main-header p { color: #558B2F; font-size: 1.1rem; }
    section[data-testid="stSidebar"] { background-color: #F1F8E9; }
    .st-emotion-cache-1qg05tj { padding: 2rem 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>\U0001F33E AgroPredict Colombia</h1><p>Optimización y Transferencia Tecnológica en el Sector Agropecuario Colombiano</p></div>', unsafe_allow_html=True)
st.divider()
