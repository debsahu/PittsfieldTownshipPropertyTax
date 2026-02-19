# Pittsfield Township Property Tax Assessment Analyzer

Streamlit web app that analyzes Pittsfield Charter Township (Washtenaw County, MI) property tax assessments using three years of official township data (2024-2026).

## What It Does

- **Parses Record Card PDFs** (OCR via tesseract) to auto-detect property details, area code, and SEV
- **Manual entry** fallback for users without a Record Card PDF
- **ECF Analysis** — shows the township's Economic Condition Factor trend and per-property ECF values
- **Comparable Sales** — arm's-length sales in your subdivision with per-property ECF columns
- **Land Value Trends** — adjustment factors and cumulative land value changes
- **Sales Coverage** — whether your area is included in the township's sales study
- **Assessment History** — year-over-year SEV/taxable value changes (from Record Card)
- **Appeal Petition Generator** — produces an L-4035-style petition with moderate, sales-based values
- **Appeal Not Recommended** guard — warns when comparable sales support the current assessment

## Architecture

```
webapp/
├── app.py              # Streamlit UI, sidebar, tabs, footer
├── analysis_engine.py  # Core computations + petition text generation
├── data_loader.py      # Loads CSV data from ../analysis/
├── rc_parser.py        # Record Card PDF → PropertyData (OCR)
├── charts.py           # Plotly chart factories
├── pyproject.toml      # Dependencies (uv)
└── .streamlit/
    ├── config.toml     # Hides deploy button
    └── secrets.toml    # APP_PASSWORD (gitignored)
```

## Data Source

All analysis data comes from official Pittsfield Township documents at [pittsfield-mi.gov/2230/Property-Assessment-Data](https://pittsfield-mi.gov/2230/Property-Assessment-Data):

- Residential Sales Analysis (2024, 2025, 2026)
- Residential ECF Analysis (2024, 2025, 2026)
- Residential Land Analysis (2024, 2025, 2026)

CSVs are stored in `../analysis/` and loaded at app startup.

## Local Development

```bash
cd webapp
uv sync
uv run streamlit run app.py
```

Requires `tesseract-ocr` installed for PDF parsing:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
apt-get install tesseract-ocr
```

## Docker

```bash
docker compose up --build
```

The app runs on port 8501. Set `APP_PASSWORD` in `.env` or as an environment variable.

## Deployment (Dokploy)

Deployed via `docker-compose.yml` on Dokploy. Dokploy handles Traefik reverse proxy and domain routing. SSL is terminated at Cloudflare (cert provider: none).

## Privacy

Uploaded Record Card PDFs are processed in memory only — never written to disk or stored. Session data is cleared when the user closes the tab.
