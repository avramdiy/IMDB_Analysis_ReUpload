import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
path = 'C:\\Users\\Ev\\Desktop\\Week 1 IMDB Analysis\\Week 1 IMDb movies.csv'  # Update with your dataset path
df = pd.read_csv(path)

# Clean column names to avoid issues with spaces or special characters
df.columns = df.columns.str.strip()

# Columns to drop before starting the API
columns_to_drop = [
    "actors", "language", "country", "date_published", "description", "director", 
    "worlwide_gross_income", "imdb_title_id", "metascore", "original_title", 
    "production_company", "title", "usa_gross_income", "writer"
]

# Drop the specified columns before starting the Flask API
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Cleaning function
def clean_data(df):
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Clean the 'budget' column by removing non-numeric characters (e.g., dollar signs, commas, and other text)
    df['budget'] = df['budget'].replace({',': '', '$': '', ' ': '', '[a-zA-Z]': ''}, regex=True)

    # Convert the 'budget' column to numeric values, setting errors to 'coerce' so that invalid values become NaN
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

    # Calculate the median of the 'budget' column, ignoring NaN values
    budget_median = df['budget'].median()

    # Replace NaN values in the 'budget' column with the median
    df['budget'].fillna(budget_median, inplace=True)

    return df

# One-hot encoding function for genres
def one_hot_encode_genres(df):
    if 'genre' in df.columns:
        genres_expanded = df['genre'].str.get_dummies(sep=',')
        df = pd.concat([df, genres_expanded], axis=1)
        df.drop(columns=['genre'], inplace=True)
    return df

# Clean the data before serving it
df = clean_data(df)
df = one_hot_encode_genres(df)

# Assuming 'avg_vote' is the target and the rest are features
X = df.drop(columns=['avg_vote'])  # Features
y = df['avg_vote']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'movie_genre_model.pkl')

print("Model trained and saved as 'movie_genre_model.pkl'")
