from typing import Optional

import requests

city_to_type = {
    "New York": "city",
    "Las Vegas": "adventure",
    "Miami": "beach",
    "Denver": "mountain",
    "Chicago": "city",
    "Goa": "beach",
    "Rockford": "cultural",
    "San Diego": "beach",
    "Los Angeles": "city",
    "Paris": "cultural",
    "Manali": "mountain",
}
def ask_ollama_for_city_type(city: str) -> str:
    prompt = f"""
        Classify the city "{city}" by its travel type (e.g., beach, mountain, cultural, adventure, city, etc.).

        Then, in one sentence, suggest a popular area or attraction a tourist should visit there.
        
        Respond in this format exactly:
        
        Type: <destination_type>
        Tip: <one-line suggestion>

    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "max_tokens": 1024,
            "top_p": 0.9,
        },
    ).json()
    return response.get("response", "")


def infer_trip_type_from_gpe(destination: Optional[str]) -> Optional[str]:
    if not destination:
        return None
    inferred_type = ask_ollama_for_city_type(destination)
    return inferred_type



def classify_budget(budget):
    if budget < 500:
        return "low", "Stay in hostels or budget Airbnbs and use public transport."
    elif 500 <= budget <= 1500:
        return "medium", "Use 3-star hotels and eat at mid-range restaurants."
    else:
        return "high", "Go for 4-star hotels, guided tours, and premium transport."

def trip_length_reasoning(days):
    if not days:
        return "Trip length not mentioned."
    if days <= 2:
        return "Short trip – focus on 1–2 major attractions."
    elif days <= 5:
        return "Medium trip – include sightseeing, food, and leisure time."
    else:
        return "Long trip – plan multiple cities or themed days."


def symbolic_reasoning(query, destination, budget=None, days=None):
    reasoning = []

    # Keyword reasoning
    inferred_type = infer_trip_type_from_gpe(destination)
    reasoning.append(f"Trip Type (inferred from destination: {destination}): {inferred_type.title()}")

    # Budget reasoning
    if budget:
        level, tip = classify_budget(budget)
        reasoning.append(f"Budget Tier: {level.title()} (${budget})\n→ {tip}")

    # Duration reasoning
    if days:
        reasoning.append(f"Trip Duration: {days} days\n→ {trip_length_reasoning(days)}")

    return "\n".join(reasoning)
