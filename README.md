# Data Engineering & Data Science Portfolio

[![Portfolio Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)]()
[![Notebooks](https://img.shields.io/badge/Notebooks-10+-blue)]()
[![Technologies](https://img.shields.io/badge/Technologies-15+-orange)]()

> **Production-grade** data engineering, machine learning, and distributed systems expertise through executable Jupyter notebooks.

---

## 🎯 Quick Start

### Option 1: View Online (Recommended for Recruiters)
1. **Start Here**: Open [INDEX.ipynb](INDEX.ipynb) in GitHub or Jupyter
2. **Navigate**: Click links to explore specific project notebooks
3. **Review**: Each notebook is self-contained with explanations and visualizations

### Option 2: Run Locally
```bash
# Install dependencies
pip install jupyter pandas numpy matplotlib seaborn scikit-learn

# Launch Jupyter
jupyter notebook

# Open INDEX.ipynb to start
```

**Note**: Notebooks are designed to run standalone. Repository code referenced in imports is included for advanced users but not required for viewing demonstrations.

---

## 📚 Portfolio Structure

### 🎯 Start Here: INDEX.ipynb
**Portfolio landing page** with:
- Navigation to all projects
- Skills visualization
- Technology stack overview
- Quick start guide

---

### 🤖 Machine Learning & Finance

#### [ML_Finance/GARCH_LSTM_Forecasting.ipynb](ML_Finance/GARCH_LSTM_Forecasting.ipynb)
**Cryptocurrency Trading System with LSTM**

- **Pipeline**: Kafka → SQLite → Feature Engineering → LSTM
- **Technologies**: PyTorch, Kafka, SQLite, Technical Analysis
- **Highlights**:
  - 50+ technical indicators (Fibonacci, RSI, ATR, Bollinger Bands)
  - Dual-output LSTM (price + volatility prediction)
  - Production trading daemon with paper/live modes
  - Data leakage prevention (temporal splits)
- **Skills**: ML Engineering, Financial Modeling, Streaming ETL
- **Repository**: [ffws_GARCHLSTM](https://github.com/anarcoiris/ffws_GARCHLSTM)

---

### 🎮 Game Theory & Reinforcement Learning

#### [ML_GameTheory/Poker_DeepRL.ipynb](ML_GameTheory/Poker_DeepRL.ipynb)
**Poker Strategy Analysis with Deep RL**

- **Approach**: Game-theoretic AI, Nash equilibrium approximation
- **Technologies**: Python, Deep Learning, Monte Carlo Simulation
- **Highlights**:
  - Self-attention for game state representation
  - Counterfactual regret minimization (CFR)
  - Monte Carlo equity calculation
  - GTO (Game Theory Optimal) strategy implementation
- **Skills**: Reinforcement Learning, Game Theory, Strategic AI
- **Repository**: [DeepGamble](https://github.com/anarcoiris/DeepGamble)

---

### 🧠 NLP & Transformers

#### [NLP/Custom_Transformer_Implementation.ipynb](NLP/Custom_Transformer_Implementation.ipynb)
**Transformer Architecture from Scratch**

- **Implementation**: Character-level language model (~10M parameters)
- **Technologies**: Python, NumPy, Deep Learning
- **Highlights**:
  - Self-attention mechanism with causal masking
  - Sinusoidal positional encoding
  - Multi-head attention (6 heads, 6 layers)
  - Sampling strategies (nucleus, top-k, temperature)
- **Skills**: Deep Learning Architecture, NLP, Low-level Implementation
- **Repository**: [MiNiLLM](https://github.com/anarcoiris/MiNiLLM)

---

### 📄 Document Processing & NLP Engineering

#### [NLP_Engineering/Document_Processing_Pipeline.ipynb](NLP_Engineering/Document_Processing_Pipeline.ipynb)
**PDF Analysis with OpenAI Integration**

- **Pipeline**: PDF Extraction → OpenAI Analysis → LaTeX Generation
- **Technologies**: PyPDF2, OpenAI API, LaTeX, scikit-learn
- **Highlights**:
  - Automated exam question extraction
  - AI-powered solution generation
  - TF-IDF clustering for similarity detection
  - Cost-optimized API batching
- **Skills**: ETL Pipelines, API Integration, Document Processing
- **Repository**: [Examn_Xterminator](https://github.com/anarcoiris/Examn_Xterminator)

---

### ☁️ Cloud Integration

#### [Cloud_Integration/Azure_Face_DataLake.ipynb](Cloud_Integration/Azure_Face_DataLake.ipynb)
**Azure Face API + Data Lake Pipeline**

- **Architecture**: Blob Storage → Face API → Data Lake → Analytics
- **Technologies**: Azure (Face API, Blob, Data Lake), Python, Streamlit
- **Highlights**:
  - Batch processing for 1000s of images
  - Face detection + attribute extraction
  - Tiered storage (hot/cool/archive)
  - Cost optimization strategies
- **Skills**: Cloud Architecture, Azure Services, Data Lake Design
- **Repository**: [FaceGUI](https://github.com/anarcoiris/FaceGUI)

---

### 🖥️ Distributed Systems

#### [Distributed_Systems/Hadoop_Docker_Setup.ipynb](Distributed_Systems/Hadoop_Docker_Setup.ipynb)
**Containerized Hadoop Cluster**

- **Stack**: Docker Compose → Hadoop → HDFS → YARN → MapReduce
- **Technologies**: Docker, Hadoop, distributed computing
- **Highlights**:
  - Multi-node cluster orchestration
  - HDFS with 3x replication
  - MapReduce job execution
  - Horizontal scaling patterns
- **Skills**: DevOps, Infrastructure-as-Code, Big Data Architecture
- **Repository**: [docker-hadoop](https://github.com/anarcoiris/docker-hadoop)

---

### 🔥 Data Engineering: Spark & Kafka

#### [PySpark/PySpak_Tutorial.ipynb](PySpark/PySpak_Tutorial.ipynb)
**PySpark Data Processing**
- Batch ETL, Spark SQL, DataFrames
- Window functions, UDFs, performance optimization

#### [PySpark/Kafka_ETL/Kafka_ETL.ipynb](PySpark/Kafka_ETL/Kafka_ETL.ipynb)
**Real-time Streaming with Kafka**
- Structured Streaming, exactly-once semantics
- Checkpointing, fault tolerance, schema evolution

---

### 📊 Causal Inference & Econometrics

#### [IIEL_notebook/notebook_IIEL.ipynb](IIEL_notebook/notebook_IIEL.ipynb)
**Spatial Econometrics & Treatment Effects**
- Panel data, fixed effects, event studies
- Tkinter GUI, Folium maps, power simulations

---

## 🛠️ Technology Stack

### Programming & Frameworks
```
Python (Advanced)    │ PyTorch │ TensorFlow │ scikit-learn
PySpark (Production) │ Pandas │ NumPy │ SQL
```

### Cloud & Infrastructure
```
Azure (Face API, Blob, Data Lake) │ Docker │ Kubernetes
Kafka │ Hadoop │ HDFS │ YARN
```

### Data Engineering
```
ETL Pipelines │ Streaming (Kafka, Spark) │ Batch Processing
Data Modeling │ Schema Design │ Data Quality
```

### Machine Learning
```
Deep Learning │ NLP (Transformers, LSTM) │ Time Series
Reinforcement Learning │ Computer Vision │ MLOps
```

---

## 📈 Portfolio Metrics

| Category | Metric | Value |
|----------|--------|-------|
| **Data Volume** | Max dataset processed | 100K+ rows |
| **ML Models** | Parameters (largest) | ~10M (Transformer) |
| **Streaming** | Throughput | Real-time (<200ms latency) |
| **Cloud** | Images processed | 1000s/batch |
| **Distributed** | Cluster nodes | 3-10 nodes (scalable) |
| **API Integration** | Services | OpenAI, Azure, Binance |

---

## 🎓 Skills Demonstrated

### Data Engineering (70%)
- ✅ ETL pipeline design and implementation
- ✅ Streaming data processing (Kafka, Spark Streaming)
- ✅ Batch processing optimization (PySpark)
- ✅ Data modeling and schema design
- ✅ Cloud data lakes and warehouses
- ✅ Distributed systems (Hadoop, HDFS)

### Data Science (20%)
- ✅ Machine learning model development
- ✅ Deep learning (PyTorch, custom architectures)
- ✅ NLP and transformers
- ✅ Time series forecasting
- ✅ Statistical modeling and causal inference

### Data Architecture (10%)
- ✅ System design for data platforms
- ✅ Infrastructure-as-Code (Docker, Kubernetes)
- ✅ Cloud architecture (Azure)
- ✅ Scalability patterns
- ✅ Cost optimization

---

## 🚀 Execution Guide

### Prerequisites
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Running Notebooks

**All notebooks are self-contained** and can run without cloning repositories:
- Demo implementations are included
- Mock data provided for external services
- Repository code is referenced for advanced users only

**To run with full repository code** (optional):
1. Clone desired repositories into respective directories:
   ```bash
   cd ML_Finance && git clone https://github.com/anarcoiris/ffws_GARCHLSTM
   cd ML_GameTheory && git clone https://github.com/anarcoiris/DeepGamble
   # etc.
   ```
2. Run notebooks - they will automatically detect and load repository code

---

## 📊 For Recruiters

### Recommended Reading Order

**1. Data Engineering Focus**:
- Start: `PySpark/Kafka_ETL/Kafka_ETL.ipynb` (streaming)
- Then: `Distributed_Systems/Hadoop_Docker_Setup.ipynb` (architecture)
- Finally: `Cloud_Integration/Azure_Face_DataLake.ipynb` (cloud)

**2. Data Science Focus**:
- Start: `ML_Finance/GARCH_LSTM_Forecasting.ipynb` (production ML)
- Then: `NLP/Custom_Transformer_Implementation.ipynb` (deep learning)
- Finally: `ML_GameTheory/Poker_DeepRL.ipynb` (advanced AI)

**3. Full-Stack Data Role**:
- Read all notebooks in order listed in INDEX.ipynb

### Time Estimates
- **Quick Review** (30 min): INDEX.ipynb + 1 notebook
- **Standard Review** (2 hours): INDEX.ipynb + 3-4 notebooks
- **Deep Dive** (1 day): All notebooks + repository code

---

## 🎯 Why This Portfolio?

### Production-Ready Code
- ✅ Error handling and validation
- ✅ Logging and monitoring
- ✅ Cost optimization
- ✅ Scalability patterns
- ✅ Documentation and testing

### Real-World Complexity
- ✅ Multi-service integration
- ✅ Distributed systems
- ✅ Cloud architecture
- ✅ Data at scale (100K+ rows)
- ✅ Production deployment patterns

### Breadth & Depth
- ✅ 10+ technologies demonstrated
- ✅ End-to-end pipelines
- ✅ ML + Engineering + Cloud
- ✅ Batch + Streaming + Real-time

---

## 📞 Repository Links

All source code available on GitHub:
- [ffws_GARCHLSTM](https://github.com/anarcoiris/ffws_GARCHLSTM) - Financial ML
- [DeepGamble](https://github.com/anarcoiris/DeepGamble) - Game Theory AI
- [MiNiLLM](https://github.com/anarcoiris/MiNiLLM) - Transformer Implementation
- [Examn_Xterminator](https://github.com/anarcoiris/Examn_Xterminator) - Document Processing
- [FaceGUI](https://github.com/anarcoiris/FaceGUI) - Azure Cloud Integration
- [docker-hadoop](https://github.com/anarcoiris/docker-hadoop) - Distributed Systems

---

## 📊 Portfolio Statistics

```
Total Notebooks:        10+
Lines of Code:          5000+ (across projects)
Technologies:           15+ frameworks/tools
Cloud Services:         3 (Azure, Binance, OpenAI)
Distributed Nodes:      Up to 10 (Hadoop cluster)
ML Model Parameters:    10M (largest model)
Data Processed:         100K+ rows (batch), Real-time (streaming)
```

---

## ✨ Recent Enhancements

- ✅ Error handling for missing dependencies
- ✅ Execution notes in all notebooks
- ✅ Enhanced inline comments for complex functions
- ✅ Standalone execution (no repository code required)
- ✅ Professional README with quick start guide

---

*This portfolio showcases enterprise-grade data engineering and data science capabilities for production environments.*

**Last Updated**: November 2025
**Status**: ✅ All notebooks executable with demo data
**Repository**: [github.com/anarcoiris/notebooks](https://github.com/anarcoiris/notebooks)
