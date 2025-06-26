# D:\food_rank_recommender_py\recommender.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2 
from psycopg2.extras import DictCursor # Explicitly import DictCursor
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Database configuration from environment variables ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 5432)), 
    'sslmode': 'require' # Or 'prefer' or 'allow' if 'require' causes issues
}

# Global DataFrame to store food items fetched from DB
df = pd.DataFrame()
tfidf_vectorizer = None
cosine_sim = None
item_indices = pd.Series() # Will be populated after loading data


# --- Function to fetch data from PostgreSQL ---
def fetch_food_items_from_db():
    print("Python Recommender: Attempting to connect to PostgreSQL database...")
    conn = None # Initialize conn to None
    try:
        # NEW: Connect using psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=DictCursor) # Use DictCursor
        
        # Adjust this query if you have more relevant columns
        # Make sure 'description' and 'category' columns exist in your 'products' table in PostgreSQL.
        cursor.execute("SELECT id, name, price, image_url, description, category FROM products")
        
        items = [dict(row) for row in cursor.fetchall()] # Convert DictRow to standard dict
        cursor.close()
        conn.close()
        print(f"Python Recommender: Successfully fetched {len(items)} food items from the database.")
        return items
    except Exception as err: 
        print(f"Python Recommender Error: Could not connect to DB or fetch data: {err}")
        print("Python Recommender: Falling back to dummy data for recommendations.")
        if conn: # Ensure connection is closed even on error
            conn.close()
        # Fallback to dummy data if DB connection fails
        # Ensure dummy data matches your actual product structure with description and category
        return [
            {"id": 1, "name": "Garden Veg Pizza", "price": 14.99, "image_url": "/images/GardernVegPizza.jpg", "description": "A delightful pizza topped with fresh garden vegetables like bell peppers, onions, and olives.", "category": "main, pizza, vegetarian, Italian"},
            {"id": 2, "name": "Cheeseburger Deluxe", "price": 11.50, "image_url": "/images/CheeseBurger.jpeg", "description": "A juicy grilled beef patty topped with melted cheddar, crisp lettuce, fresh tomato, and special sauce.", "category": "main, burger, fast food, American"},
            {"id": 3, "name": "Spicy Sushi Roll", "price": 18.00, "image_url": "/images/Sushi.jpeg", "description": "Spicy tuna and cucumber wrapped in nori and rice.", "category": "main,Japanese, Sushi"},
            {"id": 4, "name": "Chicken Tacos", "price": 9.25, "image_url": "/images/Tacos.jpeg", "description": "Crispy Shell Tacos filled with juicy Chicken and topeed with crisp lettuce and speacial sauces.", "category": "main,Tacos,Fast food, American"},
            {"id": 5, "name": "Aglio Olio Pasta", "price": 13.75, "image_url": "/images/aglio-e-olio-4.jpg", "description": "Classic Italian pasta dish with garlic, olive oil, and chili flakes.", "category": "main,Italian, Pasta, vegetarian"},
            {"id": 6, "name": "Cauliflower steak with hummus", "price": 10.00, "image_url": "/images/CauliSteak.jpeg", "description": "Thick-cut roasted cauliflower served with creamy hummus and herbs with a speacial dip.", "category": "main, vegetarian, healthy"},
            {"id": 7, "name": "Grilled Salmon", "price": 22.50, "image_url": "/images/GrilledSalmon.jpg", "description": "Perfectly grilled salmon fillet, rich in omega-3s.", "category": "main, healthy, seafood"},
            {"id": 8, "name": "Pesto Herbed Risotto", "price": 16.20, "image_url": "/images/herbedPasta.jpeg", "description": "Creamy risotto with vibrant pesto and fresh herbs.", "category": "main, Italian, vegetarian"},
            {"id": 9, "name": "Caesar Salad", "price": 8.99, "image_url": "/images/caesar-salad-1-13.jpg", "description": "Crisp romaine lettuce, croutons, parmesan cheese, and creamy Caesar dressing..", "category": "side, salad, healthy"},
            {"id": 10, "name": "Chocolate Lava Cake", "price": 7.50, "image_url": "/images/Lava-Chocolate-Cakes.jpg", "description": "Rich, warm chocolate cake with a gooey, molten center.", "category": "Dessert"},
            {"id": 11, "name": "Strawberry Shake", "price": 3.50, "image_url": "/images/milkshake.jpg", "description": "Made with rich strawberries, served with whipped cream and sauce on top ", "category": "dessert"},
            {"id": 12, "name": "Onion rings", "price": 2.50, "image_url": "/images/Onion-Rings.jpg", "description": "Crispy fried onion chips served with a hint of lemon and a dip sauce", "category": "sides, fast food, American "},
        ]


# --- Function to (re)train and save the model ---
def train_and_save_model():
    global tfidf_vectorizer, cosine_sim, df, item_indices
    print("Python Recommender: Fetching data from database for model training...")
    all_food_items = fetch_food_items_from_db()
    
    if not all_food_items:
        print("Python Recommender: No data fetched from database. Cannot train model.")
        # Clear existing models if data is empty to ensure subsequent calls also fail clearly
        tfidf_vectorizer = None
        cosine_sim = None
        df = pd.DataFrame() # Ensure df is empty
        item_indices = pd.Series()
        return

    df = pd.DataFrame(all_food_items)
    
    # Ensure 'description' and 'category' columns exist, fill with empty string if not
    if 'description' not in df.columns:
        df['description'] = ''
    if 'category' not in df.columns:
        df['category'] = ''

    df['content'] = df['name'] + ' ' + df['description'] + ' ' + df['category']
    df['content'] = df['content'].fillna('') # Handle potential missing descriptions/categories

    # Re-create item_indices based on the fresh data
    # Ensure all names are lowercase for consistent lookup
    item_indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

    print("Python Recommender: Training content-based recommendation model...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # We are no longer saving to /tmp because it's ephemeral.
    # The model will be trained in memory on every service startup.
    print("Python Recommender: Model trained successfully in memory.")


# --- Function to load the model (now primarily calls train_and_save_model) ---
def load_model():
    # This function is simplified to always re-train the model.
    # This ensures a fresh model with fresh data on every startup.
    print("Python Recommender: Initializing/re-training model on startup...")
    train_and_save_model()


# --- Recommendation Logic (using the trained model) ---
def get_content_based_recommendations(item_name, num_recommendations=3):
    global df, item_indices, tfidf_vectorizer, cosine_sim # Ensure we use the latest globals

    # Check if df is empty or models are not trained
    if df.empty or tfidf_vectorizer is None or cosine_sim is None:
        print("Python Recommender: Model not trained or data not loaded. Cannot provide content-based recommendations.")
        # If content-based fails, try to return random items from the fetched DF as a fallback
        if not df.empty:
            print("Python Recommender: Falling back to random recommendations from available data.")
            # Ensure not to return the requested item itself if it exists in DF
            available_items = [item for item in df.to_dict(orient='records') if item['name'].lower() != item_name.lower()]
            if len(available_items) >= num_recommendations:
                return random.sample(available_items, num_recommendations)
            else:
                return random.sample(available_items, len(available_items)) # Return all if less than requested
        return [] # Return empty list if no data at all


    lower_item_name = item_name.lower()
    idx = item_indices.get(lower_item_name)

    if idx is None:
        print(f"Python Recommender: Item '{item_name}' not found in our dataset for content-based recommendations. Providing random alternatives.")
        # Fallback to random if item not found, ensure it's not the requested item itself if possible
        available_items = [item for item in df.to_dict(orient='records') if item['name'].lower() != lower_item_name]
        if len(available_items) >= num_recommendations:
            return random.sample(available_items, num_recommendations)
        else:
            return random.sample(available_items, len(available_items)) # Return all if less than requested
        

    print(f"Python Recommender: Generating content-based recommendations for '{item_name}' (index {idx}).")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1] # Exclude itself and take top N

    item_indices_for_recs = [i[0] for i in sim_scores]
    recommended_items_data = df.iloc[item_indices_for_recs].to_dict(orient='records')
    print(f"Python Recommender: Raw recommendations before return: {recommended_items_data}")
    return recommended_items_data # Return full item objects


@app.route('/')
def home():
    return "Python Recommender Service is running!"

# ROUTE: /recommend-similar to be called by Node.js backend
@app.route('/recommend-similar', methods=['GET'])
def recommend_similar():
    item_name = request.args.get('item_name') # Get item_name from query parameters

    if not item_name:
        return jsonify({"error": "Missing 'item_name' query parameter"}), 400

    print(f"Python Recommender: Received request for similar items to: {item_name}")

    # Call your ML recommendation logic here
    recommendations = get_content_based_recommendations(item_name)

    # Return full item objects
    return jsonify({"recommendations": recommendations})


if __name__ == '__main__':
    # Load or train model when the application starts
    load_model() 
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000), debug=True)
