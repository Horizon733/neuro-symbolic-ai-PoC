import streamlit as st
import spacy
import json
import re
import requests
from huggingface_hub import InferenceClient
from knowledge_graph import KnowledgeGraphBuilder
from typing import List, Dict, Any, Tuple, Optional

from symbolic_reasoner import symbolic_reasoning


# Load SpaCy NER model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_lg")
    except:
        st.warning("Downloading SpaCy model. This may take a moment...")
        spacy.cli.download("en_core_web_lg")
        return spacy.load("en_core_web_lg")

# Extract GPE (Geopolitical Entities) using SpaCy
def extract_locations(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract origin and destination locations from text using SpaCy."""
    nlp = load_spacy_model()
    doc = nlp(text)
    
    # Extract all GPE entities
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    print(locations)
    
    # If we have at least two locations, assume first is origin and second is destination
    if len(locations) >= 2:
        return locations[0], locations[1]
    elif len(locations) == 1:
        return locations[0], None
    else:
        return None, None

# Extract budget using regex (similar to existing code)
def extract_budget(text: str) -> Optional[int]:
    """Extract budget amount from text using regex."""
    budget_match = re.search(r'budget\s*of\s*\$?([\d,]+)', text, re.IGNORECASE)
    if budget_match:
        return int(budget_match.group(1).replace(',', ''))
    return None

# Extract number of days using regex (similar to existing code)
def extract_days(text: str) -> Optional[int]:
    """Extract number of days from text using regex."""
    days_match = re.search(r'(\d+)\s+days?', text, re.IGNORECASE)
    if days_match:
        return int(days_match.group(1))
    return None

def format_trips_simple(trips_data):
    """
    Simple O(1) complexity formatter that converts raw trip data to readable text.
    """
    if not trips_data:
        return "No matching trips found."
    
    # Convert the entire trips data to formatted JSON
    json_str = json.dumps(trips_data, indent=2, default=str)
    
    # Create a simplified summary
    summary = f"Found {len(trips_data)} trip plans.\n\n"
    
    # Add basic interpretation instructions for the LLM
    instructions = """
    TRIP DATA INTERPRETATION GUIDE:
    - Each trip contains day_plans with activities and accommodations
    - "transportation" fields show how travelers move between locations
    - "meal" fields contain dining information
    - "attraction" fields show points of interest to visit
    - "accommodation" fields show where travelers stay

    COST ESTIMATION GUIDELINES:
    - Write all costs in plain text format (e.g., "$20 per day for 3 days = $60")
    - Present price ranges as "$100-150" (not "$100−150")
    - Do not use mathematical notation that might render incorrectly in markdown
    - DONOT use latex or any formula text, simply write the text
    - Avoid mathematical subscripts and superscripts

    The LLM should extract relevant information from this data to create a similar itinerary.
    """
    
    return summary + instructions + "\n\nJSON DATA:\n" + json_str


def query_ollama(prompt: str, model: str) -> str:
    """Query the Ollama API with a prompt and return the generated text."""
    # LM Studio
    # response = requests.post(
    #         "http://localhost:1234/v1/completions",
    #         json={
    #             "model": "qwen2.5-14b-instruct",
    #             "prompt": prompt,
    #             "stream": False,
    #             "max_tokens": 1024,
    #             "top_p": 0.9,
    #         },
    #     )
    # ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "max_tokens": 1024,
            "top_p": 0.9,
             "options": {
            "num_gpu": 1,               # Single GPU (RTX 4060)
            "num_ctx": 1600,  # Context window size
            "mirostat": 0,              # Sampling algorithm (0 = disabled)\
            "gpu_layers": 18,   # Default 32 layers on GPU for RTX 4060 (8GB VRAM)
            "f16": True,                # Use half precision for better performance on consumer GPUs
            "batch_size": 512           # Optimized for RTX 4060
        }
        },
    )

    return response.json().get("response", "No response from LLM API.")


# Streamlit UI
def main():
    st.title("🌍 AI Travel Planner")
    st.markdown("""
    Enter your travel query, and our AI will help plan your trip using real travel data!
    
    Examples:
    - "I want to travel from New York to Chicago for 5 days with a budget of $3000"
    - "Please help me plan a trip from San Francisco to Seattle"
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
        neo4j_user = st.text_input("Neo4j Username", "neo4j")
        neo4j_password = st.text_input("Neo4j Password", "password", type="password")
        ollama_model = st.selectbox("Ollama Model", ["llama3.1"])
    
    # Main query input
    query = st.text_area("What are your travel plans?", height=100)
    
    if st.button("Generate Travel Itinerary"):
        if not query:
            st.warning("Please enter your travel query!")
            return

        with st.spinner("Planning your perfect trip..."):
            # Extract locations using SpaCy
            origin, destination = extract_locations(query)
            
            # Extract other information
            days = extract_days(query)
            budget = extract_budget(query)
            
            # Display extracted information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Origin", origin if origin else "NA")
            with col2:
                st.metric("Destination", destination if destination else "NA")
            with col3:
                st.metric("Days", str(days) if days else "NA")
            with col4:
                st.metric("Budget", f"${budget:,}" if budget else "NA")
            
            # Connect to Neo4j and fetch trips
            try:
                neo4j_handler = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
                
                if origin and destination:
                    trips = neo4j_handler.fetch_trip_plans(origin, destination)
                    if not trips and origin:
                        # Fallback to origin-only search
                        st.info(f"No exact matches found for {origin} to {destination}. Showing trips from {origin} to any destination.")
                        trips = neo4j_handler.fetch_trip_plans_from_origin(origin)
                elif origin:
                    trips = neo4j_handler.fetch_trip_plans_from_origin(origin)
                else:
                    st.error("Could not detect any cities in your query. Please try again with more specific location information.")
                    return
                
                # Format trip data
                trips_context = format_trips_simple(trips)

                # Create prompt for Ollama
                prompt = f"""
                You are an expert travel planner. Create a detailed travel itinerary based on the following request:

                USER QUERY: {query}

                {trips_context}

                Based on the user query and the reference trips (if available), create a detailed day-by-day itinerary.
                Include:
                - Transportation recommendations
                - Accommodation suggestions
                - Meals and restaurants
                - Activities and attractions
                - Estimated costs where possible
                
                FORMAT REQUIREMENTS:
                1. Format the itinerary in a clear, organized way with headings for each day
                2. Write all cost calculations in plain text format (e.g., "$20 per day for 3 days = $60 total")
                3. Write price ranges with a dash (e.g., "$100-150")
                4. Do not use mathematical notation that might render incorrectly in markdown
                5. DONOT use latex or any formula text, simply write the text
                6. Avoid mathematical subscripts and superscripts
                7. Present costs in a clear table format when listing multiple expenses
                
                Format the itinerary in a clear, organized way with headings for each day.
                """
                

                # 🔍 Symbolic Reasoning
                symbolic_insight = symbolic_reasoning(query, destination=destination, budget=budget, days=days)

                # Optional display for user understanding
                with st.expander("🔎 Symbolic Reasoning Explanation"):
                    st.markdown(symbolic_insight)
                # Query Ollama
                st.subheader("Your Custom Travel Itinerary")
                itinerary = query_ollama(prompt, ollama_model)
                st.markdown(itinerary)
                
                neo4j_handler.close()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Make sure your Neo4j database is running and accessible.")

if __name__ == "__main__":
    main()