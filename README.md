# Sports Betting MVP (Streamlit)


A tiny, modular Streamlit app to:
- Load odds from CSV (or file uploader)
- Convert odds ↔ implied probabilities, remove vig (2-way)
- Compute EV and 1/4 Kelly stakes using a baseline model
- Flag 2-way arbitrage across books for the same market
- Log bets to SQLite and track bankroll/ROI


## Quickstart


```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py