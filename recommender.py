# D:\food_rank_recommender_py\recommender.py
import pandas as pd
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2 
from psycopg2 import extras # Explicitly import 'extras' or 'DictCursor' from psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
# Ensure this .env file exists in the root of your Python project for Render
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Database configuration from environment variables ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    # NEW: Ensure port is correct for PostgreSQL (5432)
    'port': int(os.getenv('DB_PORT', 5432)), 
    # NEW: Add SSL mode for Render.com PostgreSQL connection
    'sslmode': 'require' # Or 'prefer' or 'allow' if 'require' causes issues
}

# Global DataFrame to store food items fetched from DB
df = pd.DataFrame()
tfidf_vectorizer = None
cosine_sim = None
item_indices = pd.Series() # Will be populated after loading data


# --- Function to fetch data from PostgreSQL ---
def fetch_food_items_from_db():
    print("Attempting to connect to PostgreSQL database...")
    try:
        # NEW: Connect using psycopg2
        # Use a dictionary cursor for column names, like mysql.connector did
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Adjust this query if you have more relevant columns
        # Make sure 'description' and 'category' columns exist in your 'products' table in PostgreSQL.
        cursor.execute("SELECT id, name, price, image_url, description, category FROM products")
        
        items = [dict(row) for row in cursor.fetchall()] # Convert DictRow to standard dict
        cursor.close()
        conn.close()
        print(f"Successfully fetched {len(items)} food items from the database.")
        return items
    except Exception as err: # Catch a broader Exception for psycopg2 errors
        print(f"Error connecting to DB or fetching data: {err}")
        print("Falling back to dummy data for recommendations.")
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
    print("Fetching data from database for model training...")
    all_food_items = fetch_food_items_from_db()
    
    if not all_food_items:
        print("No data fetched from database. Cannot train model.")
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
    item_indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

    print("Training content-based recommendation model...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save the trained components (for faster future loads, though re-training on new data is fine)
    try:
        with open('/tmp/tfidf_vectorizer.pkl', 'wb') as f: # NEW: Use /tmp for ephemeral storage on cloud platforms
            pickle.dump(tfidf_vectorizer, f)
        with open('/tmp/cosine_sim_matrix.pkl', 'wb') as f: # NEW: Use /tmp
            pickle.dump(cosine_sim, f)
        print("Model trained and saved to /tmp.")
    except Exception as e:
        print(f"Error saving model components to /tmp: {e}")


# --- Function to load the model ---
def load_model():
    global tfidf_vectorizer, cosine_sim, df, item_indices
    # Always fetch fresh data from DB first to get up-to-date products
    print("Loading model and fetching latest data...")
    all_food_items = fetch_food_items_from_db()

    if not all_food_items:
        print("No data fetched from database. Model cannot be loaded or trained.")
        return

    df = pd.DataFrame(all_food_items)
    
    # Ensure 'description' and 'category' columns exist, fill with empty string if not
    if 'description' not in df.columns:
        df['description'] = ''
    if 'category' not in df.columns:
        df['category'] = ''

    df['content'] = df['name'] + ' ' + df['description'] + ' ' + df['category']
    df['content'] = df['content'].fillna('')

    item_indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

    try:
        # NEW: Load from /tmp, where models might have been saved if previous build created them
        with open('/tmp/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('/tmp/cosine_sim_matrix.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
        print("Model components loaded successfully from /tmp!")
    except FileNotFoundError:
        print("Model files not found in /tmp. Training new model...")
        train_and_save_model() # This will use the already fetched df
    except Exception as e:
        print(f"Error loading model from /tmp: {e}. Training new model...")
        train_and_save_model() # This will use the already fetched df


# --- Recommendation Logic (using the trained model) ---
def get_content_based_recommendations(item_name, num_recommendations=3):
    global df, item_indices # Ensure we use the latest df and item_indices

    # If df is empty (e.g., DB connection failed and dummy data wasn't returned), cannot recommend
    if df.empty or tfidf_vectorizer is None or cosine_sim is None:
        print("Model not trained or data not loaded. Cannot provide recommendations.")
        return [] # Return empty list if model/data isn't ready

    lower_item_name = item_name.lower()
    idx = item_indices.get(lower_item_name)

    if idx is None:
        print(f"Item '{item_name}' not found in our dataset for content-based recommendations. Providing random alternatives.")
        # Fallback to random if item not found, ensure it's not the requested item itself if possible
        # Ensure there are enough items to sample
        available_items = [item for item in df.to_dict(orient='records') if item['name'].lower() != lower_item_name]
        if len(available_items) >= num_recommendations:
            return random.sample(available_items, num_recommendations)
        else:
            return random.sample(available_items, len(available_items)) # Return all if less than requested
        

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1] # Exclude itself and take top N

    item_indices_for_recs = [i[0] for i in sim_scores]
    return df.iloc[item_indices_for_recs].to_dict(orient='records') # Return full item objects


@app.route('/')
def home():
    return "Python Recommender Service is running!"

# NEW ROUTE: /recommend-similar to be called by Node.js backend
# Using GET method and query parameters for simplicity for Node.js fetch
@app.route('/recommend-similar', methods=['GET'])
def recommend_similar():
    item_name = request.args.get('item_name') # Get item_name from query parameters

    if not item_name:
        return jsonify({"error": "Missing 'item_name' query parameter"}), 400

    print(f"Python Recommender received request for similar items to: {item_name}")

    # Call your ML recommendation logic here
    recommendations = get_content_based_recommendations(item_name)

    # Return full item objects
    return jsonify({"recommendations": recommendations})


if __name__ == '__main__':
    # Load or train model when the application starts
    load_model()
    # NEW: For local development, use Flask's default development server
    # For Render deployment, Gunicorn will start the app
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000), debug=True)