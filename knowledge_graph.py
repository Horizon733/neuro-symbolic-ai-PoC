import ast
from neo4j import GraphDatabase
from datasets import load_dataset
import sys

class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        """Initialize the Neo4j driver."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
    
    def build_graph_from_travel_dataset(self):
        """
        Load the osunlp/TravelPlanner dataset from Hugging Face and build a knowledge graph.
        For each record, we create:
          - City nodes for origin and destination.
          - A TripPlan node.
          - DayPlan nodes (from the annotated plan) with related nodes for transportation,
            meals, attractions, and accommodations.
          - ReferenceInfo nodes for any reference information.
        """
        print("Loading osunlp/TravelPlanner dataset from Hugging Face...")
        try:
            dataset = load_dataset("osunlp/TravelPlanner", "train")
        except Exception as e:
            sys.exit(f"Error loading dataset: {e}")
        
        # Use a chosen split (here, "train") – adjust if needed.
        split_name = "train"
        if split_name not in dataset:
            sys.exit(f"Split '{split_name}' not found. Available splits: {list(dataset.keys())}")
        
        print("Building knowledge graph from dataset...")
        with self.driver.session() as session:
            for record in dataset[split_name]:
                self.process_record(session, record)
        print("Knowledge graph build complete.")
    
    def process_record(self, session, record):
        # Extract key fields from the record.
        org = record.get("org", "Unknown")
        dest = record.get("dest", "Unknown")
        days = record.get("days", 0)
        visiting_city_number = record.get("visiting_city_number", 0)
        date = record.get("date", "[]")
        people_number = record.get("people_number", 0)
        local_constraint = record.get("local_constraint", "{}")
        budget = record.get("budget", 0)
        query_text = record.get("query", "")
        level = record.get("level", "unknown")
        annotated_plan = record.get("annotated_plan", "[]")
        reference_information = record.get("reference_information", "[]")
        
        # Create (or merge) City nodes for origin and destination.
        session.write_transaction(self._merge_city, org)
        session.write_transaction(self._merge_city, dest)
        
        # Create the TripPlan node and get its internal ID.
        session.write_transaction(
            self._create_trip_plan,
            org, dest, days, visiting_city_number, date, people_number,
            local_constraint, budget, query_text, level, annotated_plan, reference_information
        )
    
    @staticmethod
    def _merge_city(tx, city_name):
        """Merge a City node with the given name."""
        tx.run(
            "MERGE (c:City {name: $city_name})",
            city_name=city_name
        )
    
    @staticmethod
    def _create_trip_plan(tx, org, dest, days, visiting_city_number, date,
                          people_number, local_constraint, budget, query_text,
                          level, annotated_plan, reference_information):
        """
        Create a TripPlan node with its properties and relate it to origin/destination cities.
        Then process the annotated plan and reference information.
        """
        # Create the TripPlan node.
        result = tx.run(
            """
            CREATE (tp:TripPlan {
                org: $org,
                dest: $dest,
                days: $days,
                visiting_city_number: $visiting_city_number,
                date: $date,
                people_number: $people_number,
                local_constraint: $local_constraint,
                budget: $budget,
                query: $query_text,
                level: $level,
                annotated_plan: $annotated_plan,
                reference_information: $reference_information
            })
            RETURN id(tp) AS tp_id
            """,
            org=org,
            dest=dest,
            days=days,
            visiting_city_number=visiting_city_number,
            date=date,
            people_number=people_number,
            local_constraint=local_constraint,
            budget=budget,
            query_text=query_text,
            level=level,
            annotated_plan=annotated_plan,
            reference_information=reference_information
        )
        record = result.single()
        tp_id = record["tp_id"]
        
        # Create relationships from the TripPlan node to the origin and destination cities.
        tx.run(
            """
            MATCH (tp:TripPlan) WHERE id(tp) = $tp_id
            MATCH (c1:City {name: $org})
            MATCH (c2:City {name: $dest})
            MERGE (tp)-[:ORIGIN]->(c1)
            MERGE (tp)-[:DESTINATION]->(c2)
            """,
            tp_id=tp_id,
            org=org,
            dest=dest
        )
        
        # Process the annotated_plan field to create DayPlan and related nodes.
        try:
            day_plans = ast.literal_eval(annotated_plan)
        except Exception as e:
            day_plans = []
        
        if isinstance(day_plans, list):
            for day_plan in day_plans:
                if isinstance(day_plan, dict) and day_plan:
                    KnowledgeGraphBuilder._create_day_plan(tx, tp_id, day_plan)
        
        # Process the reference_information field.
        try:
            ref_infos = ast.literal_eval(reference_information)
        except Exception as e:
            ref_infos = []
        
        if isinstance(ref_infos, list):
            for ref in ref_infos:
                if isinstance(ref, dict) and "Description" in ref and "Content" in ref:
                    KnowledgeGraphBuilder._create_reference_info(tx, tp_id, ref)
    
    @staticmethod
    def _create_day_plan(tx, tp_id, day_plan):
        """
        Create a DayPlan node for a specific day's plan and create related nodes for transportation,
        meals, attractions, and accommodations.
        """
        day_number = day_plan.get("days", None)
        current_city = day_plan.get("current_city", "")
        transportation = day_plan.get("transportation", "")
        breakfast = day_plan.get("breakfast", "")
        attraction = day_plan.get("attraction", "")
        lunch = day_plan.get("lunch", "")
        dinner = day_plan.get("dinner", "")
        accommodation = day_plan.get("accommodation", "")
        
        # Create the DayPlan node and attach it to the TripPlan.
        result = tx.run(
            """
            MATCH (tp:TripPlan) WHERE id(tp) = $tp_id
            CREATE (dp:DayPlan {day: $day_number, current_city: $current_city})
            MERGE (tp)-[:HAS_DAY_PLAN]->(dp)
            RETURN id(dp) AS dp_id
            """,
            tp_id=tp_id,
            day_number=day_number,
            current_city=current_city
        )
        record = result.single()
        dp_id = record["dp_id"]
        
        # If current_city is provided, merge it as a City node and create a relationship.
        if current_city and current_city.strip() != "-":
            tx.run(
                "MERGE (c:City {name: $current_city})",
                current_city=current_city
            )
            tx.run(
                """
                MATCH (dp:DayPlan) WHERE id(dp) = $dp_id
                MATCH (c:City {name: $current_city})
                MERGE (dp)-[:IN_CITY]->(c)
                """,
                dp_id=dp_id,
                current_city=current_city
            )
        
        # For each related field, if there is valid data (not '-' or empty), create a related node.
        if transportation and transportation.strip() != "-":
            KnowledgeGraphBuilder._create_related_node(tx, dp_id, transportation, "Transportation")
        if breakfast and breakfast.strip() != "-":
            KnowledgeGraphBuilder._create_related_node(tx, dp_id, breakfast, "Meal", meal_type="Breakfast")
        if lunch and lunch.strip() != "-":
            KnowledgeGraphBuilder._create_related_node(tx, dp_id, lunch, "Meal", meal_type="Lunch")
        if dinner and dinner.strip() != "-":
            KnowledgeGraphBuilder._create_related_node(tx, dp_id, dinner, "Meal", meal_type="Dinner")
        if attraction and attraction.strip() != "-":
            KnowledgeGraphBuilder._create_related_node(tx, dp_id, attraction, "Attraction")
        if accommodation and accommodation.strip() != "-":
            KnowledgeGraphBuilder._create_related_node(tx, dp_id, accommodation, "Accommodation")
    
    @staticmethod
    def _create_related_node(tx, dp_id, value, label, meal_type=None):
        """
        Create a related node (e.g. Transportation, Meal, Attraction, Accommodation)
        and link it to the DayPlan node.
        Optionally, include a 'meal_type' property if the label is Meal.
        """
        if meal_type:
            tx.run(
                f"""
                MATCH (dp:DayPlan) WHERE id(dp) = $dp_id
                CREATE (n:{label} {{value: $value, meal_type: $meal_type}})
                MERGE (dp)-[:HAS_{label.upper()}]->(n)
                """,
                dp_id=dp_id,
                value=value,
                meal_type=meal_type
            )
        else:
            tx.run(
                f"""
                MATCH (dp:DayPlan) WHERE id(dp) = $dp_id
                CREATE (n:{label} {{value: $value}})
                MERGE (dp)-[:HAS_{label.upper()}]->(n)
                """,
                dp_id=dp_id,
                value=value
            )
    
    @staticmethod
    def _create_reference_info(tx, tp_id, ref):
        """
        Create a ReferenceInfo node from reference data and link it to the TripPlan.
        """
        description = ref.get("Description", "")
        content = ref.get("Content", "")
        tx.run(
            """
            MATCH (tp:TripPlan) WHERE id(tp) = $tp_id
            CREATE (ri:ReferenceInfo {description: $description, content: $content})
            MERGE (tp)-[:HAS_REFERENCE]->(ri)
            """,
            tp_id=tp_id,
            description=description,
            content=content
        )
    
    def fetch_trip_plans(self, origin, destination):
        """
        Query the Neo4j knowledge graph for TripPlan nodes
        that have an ORIGIN city matching 'origin' and a DESTINATION city matching 'destination'.
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._get_trip_plans, origin, destination)
            # Convert each Neo4j node to a dict for easier handling.
            return [dict(tp) for tp in result]

    
    def fetch_trip_plans_from_origin(self, origin):
        """
        Query the Neo4j knowledge graph for TripPlan nodes
        that have an ORIGIN city matching 'origin'.
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._get_trip_plans_from_origin, origin)
            # Convert each Neo4j node to a dict for easier handling.
            return [dict(tp) for tp in result]


    @staticmethod
    def _get_trip_plans(tx, origin, destination):
        query = (
            "MATCH (t:TripPlan) "
            "WHERE toLower(t.org) = toLower($origin) "
            "  AND toLower(t.dest) = toLower($destination) "
            "RETURN t.org AS org, t.dest AS dest, t.days AS days, t.date AS date, "
            "       t.people_number AS people_number, t.budget AS budget, "
            "       t.query AS query_text, t.level AS level, t.annotated_plan AS annotated_plan, "
            "       t.reference_information AS reference_information"
        )
        result = tx.run(query, origin=origin, destination=destination)
        return [record.data() for record in result]
    

    @staticmethod
    def _get_trip_plans_from_origin(tx, origin):
        query = (
            "MATCH (t:TripPlan) "
            "WHERE toLower(t.org) = toLower($origin) "
            "RETURN t.org AS org, t.dest AS dest, t.days AS days, t.date AS date, "
            "       t.people_number AS people_number, t.budget AS budget, "
            "       t.query AS query_text, t.level AS level, t.annotated_plan AS annotated_plan, "
            "       t.reference_information AS reference_information"
        )
        result = tx.run(query, origin=origin)
        return [record.data() for record in result]

