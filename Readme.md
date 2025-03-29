# 🧠🌍 AI Travel Planner: Neurosymbolic Conversational Assistant

A powerful **Conversational AI** that blends **neural LLMs**, **symbolic reasoning**, and a **knowledge graph** to generate personalized, intelligent travel itineraries from natural language queries.

Built using:
- 🤖 **Ollama** + **LangChain** for LLM-powered text generation
- 🧠 **Symbolic rules** for logic-based reasoning
- 🧬 **Neo4j** for knowledge graph trip storage and queries
- 🧵 **Streamlit** for a beautiful interactive UI
- 🌍 **SpaCy** for NLP and entity extraction
- 📊 **Hugging Face Datasets** for real-world trip data

---

## 📦 Dataset

This app is powered by the [`osunlp/TravelPlanner`](https://huggingface.co/datasets/osunlp/TravelPlanner) dataset available on Hugging Face.

It contains:
- Realistic trip planning queries
- Origin/destination cities
- Duration, budget, constraints
- Annotated plans: attractions, transportation, food, stay info
- Reference information about cities, flights, hotels, restaurants

Used as the **core knowledge base**, loaded into **Neo4j** as a symbolic graph.

---

## 💡 Key Features

- 🧾 Understands natural queries like:  
  _"Plan a 3-day trip from Kansas City to New York with $1500 budget"_
  
- 🧠 Extracts **origin**, **destination**, **budget**, **duration** using **SpaCy + regex**
- 📖 Applies **symbolic reasoning** (rules based on city, budget, trip length)
- 🔄 Fetches structured trip data from **Neo4j** (used as retrieval context)
- 🤖 Uses **Ollama (LLaMA 3.1 or other local models)** to generate full itineraries
- 🧩 Combines both neural (LLM) and symbolic (rules + knowledge graph) reasoning
- 🧵 Built entirely in **Streamlit**

---

## 🔧 How It Works

1. **User inputs** a free-text travel query
2. **SpaCy** extracts origin, destination, days, budget
3. **Symbolic module** infers trip type and travel advice using rules
4. **Neo4j** is queried for relevant real-world trip plans
5. **LLM (Ollama)** receives all structured info and generates a custom itinerary
6. Itinerary includes: **transportation, accommodation, meals, attractions, and estimated costs**

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/neurosymbolic-travel-planner.git
cd neurosymbolic-travel-planner
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Start Ollama
Make sure you have Ollama installed and a model like llama3 or qwen2.5 downloaded.
```bash 
ollama run llama3.1
```

### 4. Load Neo4j and Import TravelPlanner Data
Ensure Neo4j is running on your machine or remote.

Create a .env file or input connection details from the sidebar.

Use provided scripts (or your own) to load the `osunlp/TravelPlanner` dataset into Neo4j as a trip graph.

### 5. Run the Streamlit App
```bash
streamlit run travel_planner.py
```

## 🧠 Symbolic Component
We apply structured, rule-based logic for:
- Inferring trip types (e.g., city, beach, mountain) based on destination
- Budget tier classification
- Suggesting travel advice based on trip length
- Fallback logic when Neo4j returns no trips
- Built with flexible Python rules, and optionally enriched via Neo4j or LLM fallback.

## 🤖 Neural Component (LLM via Ollama)
- Uses Ollama to run models like llama3.1, qwen, or mistral
- Prompt includes:
  - User query
  - Extracted symbols
  - Symbolic insights
  - Retrieved trips from Neo4j
- Generates a day-by-day itinerary with natural language and cost summaries

## Demo


https://github.com/user-attachments/assets/9aa02a94-742a-4f74-9169-107f3ea2b86e


