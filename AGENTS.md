# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python/Streamlit codebase containing financial analytics dashboards:
- **Primary app**: `bonus_abuse_detection.py` — Forex/CFD broker bonus abuse detection dashboard (the one configured to auto-launch)
- **Secondary apps**: Market forecasting dashboards (`streamlit_forecast_final.py`, `oil_forecast_dashboard.py`, etc.)

No databases, Docker services, or build steps are required. Pure Python + Streamlit.

### Running the app

```bash
streamlit run bonus_abuse_detection.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8501 --server.headless true
```

The app serves on port 8501. Sample data is built into the app (no external data sources required for the primary dashboard).

### Linting

```bash
flake8 bonus_abuse_detection.py --max-line-length=120
```

The codebase has pre-existing style issues (long lines, unused imports). These are not regressions.

### Key caveats

- `streamlit` installs to `~/.local/bin` — ensure `PATH` includes `$HOME/.local/bin`.
- The market forecasting apps (`app (1).py`, `streamlit_forecast_with_arima.py`, etc.) require `statsmodels` which is NOT in `requirements.txt`. Install separately if needed: `pip install statsmodels`.
- Some app files (`app (1).py`, `app (2).py`) require `prophet` which is also not in `requirements.txt`.
- MetaTrader 5 integration in `bonus_abuse_detection.py` is Windows-only and optional — it gracefully handles the import failure.
- Many files are iterative versions exported from Google Colab. The canonical deployable app is `bonus_abuse_detection.py`.
