# AeroPulse: AI-Powered Ambient Air Quality Forecasting & Policy Intelligence System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-228B22.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-29%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

> **"Understand your air. Make better decisions."**  
> AeroPulse is an end-to-end ambient air quality intelligence platform combining strictly leak-free machine learning forecasting across 6 major Indian metropolitan airsheds with a statutory policy retrieval-augmented generation (RAG) assistant for citizens and environmental policymakers.

---

## 1. System Overview

AeroPulse bridges the gap between high-dimensional environmental time-series modeling and actionable public health policy. The platform unifies:
1. **Empirical Multi-City Forecasting**: Leakage-safe, prior-only sequence and tabular machine learning models (XGBoost, Bi-LSTM, Bi-GRU, Dilated TCN, Extra Trees, and Heterogeneous Stacking Ensembles) trained across 6 major Indian cities (Delhi, Gurugram, Bengaluru, Chennai, Hyderabad, Patna).
2. **Policy Intelligence RAG Engine**: A hybrid dense-sparse (FAISS + BM25) retrieval pipeline grounded in official statutory frameworks including the Graded Response Action Plan (GRAP 2024), National Clean Air Programme (NCAP), Air (Prevention and Control of Pollution) Act 1981, and CPCB National Ambient Air Quality Standards (NAAQS).
3. **Dual-Mode Interactive Application**:
   - **Public AI Assistant**: Conversational citizen interface for daily health advisories, outdoor safety guidance, and statutory awareness.
   - **Government & Analytics Intelligence**: Real-time forecast evaluation, model ablation comparison, criteria pollutant trends, and statutory intervention simulation.

---

## 2. Core Capabilities

- **Strict Prior-Only Causal Formulation**: Eliminates contemporaneous leakage by enforcing $X(t-14, \dots, t-1) \to \text{AQI}(t)$ with shifted rolling statistics and split-isolated scalers.
- **Heterogeneous Stacking Ensembles**: Evaluates multi-paradigm combinations (SE-1: XGBoost + Bi-LSTM + TCN; SE-2: XGBoost + Bi-GRU + TCN) fused via an Extra Trees meta-learner trained exclusively on out-of-fold (OOF) cross-validation predictions.
- **Multi-City Airshed Generalization**: Multi-city spatial coverage across Northern, Eastern, and Southern Indian meteorological regimes.
- **Hybrid Sparse-Dense Policy Retrieval**: Reciprocal Rank Fusion (RRF) combining BM25 keyword matching with dense sentence embeddings (`all-MiniLM-L6-v2`) over authoritative Indian environmental statutes.
- **Multi-Tiered LLM Provider Architecture**: Seamless runtime failover across NVIDIA NIM (`openai/gpt-oss-20b`), Google Gemini, OpenAI (`gpt-4o-mini`), and a deterministic offline grounded template engine for zero-dependency local execution.
- **Robust Guardrails & Intent Routing**: 9-class deterministic intent classifier with built-in prompt-injection neutralization, domain boundary enforcement, and statutory citation verification.

---

## 3. High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AEROPULSE PLATFORM                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────────────┐          ┌───────────────────────────────┐   │
│   │    Public AI Assistant    │          │  Government Analytics Portal  │   │
│   │ (Conversational Citizens) │          │    (Policymakers / Experts)   │   │
│   └─────────────┬─────────────┘          └───────────────┬───────────────┘   │
│                 │                                        │                   │
│                 └──────────────────┬─────────────────────┘                   │
│                                    ▼                                         │
│                      FastAPI Application Gateway                             │
│                  (/api/chat, /api/predict, /api/policy)                      │
│                                    │                                         │
│         ┌──────────────────────────┴──────────────────────────┐              │
│         ▼                                                     ▼              │
│   ┌──────────────────────────────┐              ┌──────────────────────────┐ │
│   │    ML Forecasting Engine     │              │   Policy RAG Engine      │ │
│   ├──────────────────────────────┤              ├──────────────────────────┤ │
│   │ • 65 Tabular Features        │              │ • Chunked Statutory Docs │ │
│   │ • 10-Channel 3D Sequences    │              │ • BM25 + FAISS Dense     │ │
│   │ • XGBoost / Extra Trees      │              │ • Reciprocal Rank Fusion │ │
│   │ • PyTorch Bi-LSTM / Bi-GRU   │              │ • Citation Verifier      │ │
│   │ • Dilated Temporal Conv(TCN) │              │ • Prompt Guardrails      │ │
│   │ • Stacking Meta-Learners     │              │ • Intent Classifier      │ │
│   └──────────────┬───────────────┘              └────────────┬─────────────┘ │
│                  │                                           │               │
│                  └─────────────────────┬─────────────────────┘               │
│                                        ▼                                     │
│                       Multi-Tier LLM Provider Router                         │
│             [ NVIDIA NIM  →  Gemini  →  OpenAI  →  Deterministic Offline ]   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```text
AQI_Research_Project/
├── app/
│   ├── main.py                     # FastAPI server, REST routes & lifecycle
│   ├── static/                     # Static CSS/JS assets
│   └── templates/
│       └── index.html              # Unified Public Assistant & Gov Dashboard UI
├── data/
│   ├── cache/                      # Precomputed RAG search indices (rag_index.pkl)
│   ├── interim/                    # Merged multi-city daily series & weather
│   ├── policy_documents/           # Statutory corpus (GRAP 2024, NCAP, Air Act)
│   ├── processed/                  # Scaled dataset bundles (dataset_bundle.pkl, research_bundle.pkl)
│   └── raw/                        # CPCB raw observations (city_day.csv, weather_daily.csv)
├── experiments/
│   ├── research/                   # Multi-city research benchmark & SHAP scripts
│   │   ├── run_full_pipeline.py    # Full multi-city pipeline orchestrator
│   │   ├── run_training.py         # Multi-city model training & evaluation
│   │   ├── run_shap_analysis.py    # SHAP feature importance analysis
│   │   └── generate_shap_final.py  # SHAP publication figures generator
│   └── run_full_prior_experiments.py # Delhi NCR baseline experiment runner
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Comprehensive EDA & visualization notebook
├── results/
│   ├── figures/                    # Publication charts (model comparison, residuals, etc.)
│   ├── metrics/                    # Saved evaluation metrics and ablation tables
│   ├── predictions/                # Out-of-sample prediction artifacts
│   ├── reports/                    # Statistical hypothesis tests & significance reports
│   └── research/                   # Multi-city research figures, tables & SHAP artifacts
├── src/
│   ├── data/                       # Ingestion, validation, and interpolation modules
│   ├── features/                   # Prior-only feature engineering & sequence builders
│   ├── models/                     # PyTorch deep sequence models & gradient boosting
│   ├── rag/                        # Complete Policy RAG & Conversational Assistant
│   │   ├── assistant.py            # Public conversational assistant & intent router
│   │   ├── chunker.py              # Semantic markdown & text chunking pipeline
│   │   ├── document_loader.py      # Multi-format statutory document loader
│   │   ├── evaluator.py            # Automated RAG evaluation & failure mode auditing
│   │   ├── generator.py            # Multi-provider LLM response generator
│   │   ├── pipeline.py             # End-to-end Policy RAG coordinator
│   │   ├── retriever.py            # Hybrid BM25 + FAISS dense retriever
│   │   └── schemas.py              # Pydantic validation schemas
│   ├── stacking/                   # TimeSeriesSplit OOF stacking meta-learner
│   └── utils/                      # Project logging and configuration helpers
├── tests/
│   ├── test_api.py                 # REST endpoint integration tests
│   ├── test_assistant.py           # Public assistant intent & safety unit tests
│   ├── test_leakage.py             # Temporal leakage & causal integrity tests
│   ├── test_rag.py                 # RAG retrieval, citation & guardrail tests
│   └── test_stacking.py            # OOF stacking validation tests
├── PROJECT_LOG.md                  # Comprehensive engineering & research decision log
├── pytest.ini                      # Pytest runner configuration
├── requirements.txt                # Production & research dependencies
└── README.md                       # Complete platform documentation
```

---

## 5. Technology Stack

- **Core Runtime**: Python 3.12
- **Machine Learning & Deep Learning**:
  - `xgboost` (2.0+) — Gradient boosted decision trees
  - `scikit-learn` (1.4+) — Extra Trees, preprocessing, time-series splits
  - `torch` (2.2+) — PyTorch for Bi-LSTM, Bi-GRU, and Dilated TCN sequence models
  - `shap` (0.44+) — TreeExplainer & DeepExplainer game-theoretic interpretability
- **Retrieval-Augmented Generation (RAG)**:
  - `sentence-transformers` (`all-MiniLM-L6-v2`) — Dense semantic embeddings
  - `faiss-cpu` — Dense vector similarity index
  - `rank_bm25` — Lexical sparse BM25 scoring
- **Web & API Framework**:
  - `fastapi` & `uvicorn` — Asynchronous REST API
  - `jinja2` — Dynamic template rendering
  - `pydantic` — Strict request/response payload validation
  - `Bootstrap 5` & `Chart.js` — Responsive UI and interactive timeseries visualization

---

## 6. Installation & Setup

### 6.1 Clone the Repository
```bash
git clone https://github.com/Sohamk22/aeropulse-ai.git
cd aeropulse-ai
```

### 6.2 Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 6.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Environment Variables & LLM Configuration

AeroPulse is designed to operate seamlessly both **online** (with cloud LLM providers) and **offline** (100% deterministic fallback without API keys).

To enable live LLM synthesis, create a `.env` file or export environment variables:

```bash
# Optional: NVIDIA NIM API (Primary live provider)
export NVIDIA_API_KEY="nvapi-..."
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_MODEL="openai/gpt-oss-20b"

# Optional: Google Gemini API (Secondary live provider)
export GEMINI_API_KEY="AIzaSy..."
export GEMINI_MODEL="gemini-1.5-flash"

# Optional: OpenAI API (Tertiary live provider)
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
```

### Provider Fallback Priority
1. **NVIDIA NIM** (if `NVIDIA_API_KEY` is present)
2. **Google Gemini** (if `GEMINI_API_KEY` is present)
3. **OpenAI** (if `OPENAI_API_KEY` is present)
4. **Deterministic Offline Grounded Engine** (default fallback when no keys are configured)

---

## 8. Running the Application

### 8.1 Launch the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```text
http://localhost:8000
```

### 8.2 Interacting with the Application
- **Public Assistant View**: Ask natural-language questions such as:
  - *"Is it safe for a morning jog today?"*
  - *"What GRAP Stage 3 emergency actions are in effect?"*
  - *"Why is Delhi's AQI spiking this week?"*
  - *"What legal authority empowers the CPCB to shut down polluting industries?"*
- **Government Analytics View**: Switch to the expert dashboard to view:
  - Model point forecasts vs actuals (XGBoost, SE-1, SE-2, Extra Trees)
  - 14-day criteria pollutant trajectories (PM2.5, PM10, NO2, SO2, CO, O3)
  - Statutory policy advisory synthesis with active citation links
  - Cross-model residual distributions and performance benchmarks

---

## 9. Automated Testing Suite

The repository includes a comprehensive test suite covering data leakage, model stacking, REST APIs, RAG retrieval, guardrails, and conversational intent routing:

```bash
pytest tests/ -v
```

### Test Coverage Highlights (29/29 Passing)
- `tests/test_leakage.py`: Verifies zero contemporaneous feature leakage, strict 1-step rolling window shifting, and split-isolated scalers.
- `tests/test_stacking.py`: Validates 5-fold TimeSeriesSplit out-of-fold execution and Extra Trees meta-learner isolation.
- `tests/test_api.py`: Tests `/api/predict`, `/api/insights`, `/api/available_dates`, and `/api/metrics`.
- `tests/test_rag.py`: Verifies document chunking, hybrid BM25+FAISS retrieval, citation accuracy, unanswerable query refusal, and prompt injection defense.
- `tests/test_assistant.py`: Verifies intent routing across all 9 intent categories, response structure, and domain boundary enforcement.

---

## 10. Machine Learning & Multi-City Research Findings

### 10.1 Multi-City Empirical Benchmark
The research pipeline evaluated standalone and stacked models across 6 major Indian cities (Delhi, Gurugram, Bengaluru, Chennai, Hyderabad, Patna) representing 7,843 total daily observations under strict out-of-sample temporal testing (1,770 test samples):

| Model Architecture | Model Paradigm | Test RMSE ↓ | Test MAE ↓ | Test R² ↑ | Multi-City Stability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Standalone)** | Gradient Boosted Trees | **28.10** | **18.94** | **0.9225** | High |
| **Extra Trees (Standalone)** | Randomized Forest Ensemble | **28.37** | **19.12** | **0.9210** | High |
| **SE-1 (XGB + BiLSTM + TCN)** | Stacked Meta-Learner (ET) | **30.81** | **20.45** | **0.9068** | Moderate |
| **SE-2 (XGB + BiGRU + TCN)** | Stacked Meta-Learner (ET) | **42.53** | **28.74** | **0.8224** | Moderate |
| **Dilated TCN (Standalone)** | Deep Temporal Convolution | **45.55** | **31.20** | **0.7963** | Moderate |
| **Bi-GRU (Standalone)** | Recurrent Sequence Model | 108.15 | 78.40 | -0.1482 | Low |
| **Bi-LSTM (Standalone)** | Recurrent Sequence Model | 109.69 | 79.15 | -0.1810 | Low |

### 10.2 Key Research Insights
1. **Tree Ensembles Excel on Prior Features**: Gradient boosted and randomized decision trees decisively outperformed standalone deep recurrent sequence models on daily atmospheric tabular data.
2. **SE-1 Outperforms SE-2**: In multi-city stacking, the combination of Bi-LSTM and TCN temporal representations with XGBoost (SE-1) provided superior meta-learner regularization compared to Bi-GRU (SE-2).
3. **Severe Event Complementarity**: While tree models achieved optimal mean metric scores, deep sequence models provided orthogonal inductive bias during extreme winter spike events ($Z > 3.0$), dampening peak underestimation error.

---

## 11. Policy RAG & Intelligence System

### 11.1 Indexed Statutory Corpus
The policy knowledge base indexes key Indian environmental regulations located in `data/policy_documents/`:
- **GRAP 2024**: Graded Response Action Plan for Delhi-NCR (Stages I–IV triggers, dust control, vehicular restrictions, industrial closures).
- **NCAP Guidelines**: National Clean Air Programme reduction targets, city action plans, and monitoring protocols.
- **Air Act 1981**: Statutory powers of State and Central Pollution Control Boards under Sections 19, 21, and 31A.
- **CPCB NAAQS Standards**: National Ambient Air Quality Standards for criteria pollutants ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$).

### 11.2 Hybrid Retrieval & Verification
1. **Semantic Chunking**: Documents are split into coherent structural chunks with metadata tags (Act, Section, Severity Threshold, Target Pollutant).
2. **Reciprocal Rank Fusion**: Merges lexical BM25 rankings with dense vector cosine similarity to ensure precise statutory section matching.
3. **Grounded Synthesis & Defense**: LLM synthesis enforces strict statutory citations (`[Doc: Section]`) and rejects adversarial prompt injection attempts.

---

## 12. API Reference

### Forecast Endpoint
`POST /api/predict`
```json
{
  "date": "2020-01-15"
}
```
**Response**: Point forecasts for XGBoost, SE-1, SE-2, Extra Trees, consensus AQI, and 14-day prior historical pollutant context.

### Public Assistant Chat Endpoint
`POST /api/chat` or `POST /api/assistant/query`
```json
{
  "query": "Is it safe for children to play outside this afternoon?",
  "date": "2020-01-15",
  "top_k": 5
}
```
**Response**: Intent-classified response, synthesized advisory, statutory policy citations, and risk assessment badge.

### Policy Analysis Endpoint
`POST /api/policy/analyze`
```json
{
  "date": "2020-01-15",
  "top_k": 5
}
```
**Response**: Statutory severity category, triggered regulatory stages, mandated government interventions, and citation list.

---

## 13. Current Limitations & Future Roadmap

1. **Multi-Horizon Forecasting**: Currently models 1-day ahead ($\tau = 1$). Expanding to multi-step direct autoregressive forecasting ($\tau \in \{1, 3, 7\}$ days) is planned.
2. **Spatial Advection Modeling**: Future iterations will integrate Spatio-Temporal Graph Neural Networks (ST-GNN) to explicitly model inter-city atmospheric transport and wind advection vectors.
3. **Numerical Weather Prediction (NWP) Forcing**: Integration with real-time operational meteorological forecasts (GFS / ECMWF) will supplement historical meteorological lags.

---

## 14. License & Attribution

This project is developed under the MIT License. Atmospheric observational datasets are provided via the Central Pollution Control Board (CPCB) India Open Government Data platform and ECMWF ERA5 Atmospheric Reanalysis.
