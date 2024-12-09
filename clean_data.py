from flask import Flask, jsonify, request, render_template, send_file, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'), static_folder=os.path.join(os.getcwd(), 'static'))

# Define the path to your dataset
path = 'C:\\Users\\Ev\\Desktop\\Week 1 IMDB Analysis\\Week 1 IMDb movies.csv'  # Update with your dataset path

# Load the dataset
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
# Adjust the one-hot encoding function to handle the missing columns
def one_hot_encode_genres(df):
    if 'genre' in df.columns:
        # One-hot encode genres
        genres_expanded = df['genre'].str.get_dummies(sep=',')
        df = pd.concat([df, genres_expanded], axis=1)
        df.drop(columns=['genre'], inplace=True)  # Drop the original genre column
    return df


# Clean the data before serving it
df = clean_data(df)
df = one_hot_encode_genres(df)

# Function to generate the bar chart
def create_genre_ratings_chart(df):
    genre_columns = df.select_dtypes(include=[np.number]).columns.drop('avg_vote')
    average_ratings_by_genre = {
        genre: (df[genre] * df['avg_vote']).sum() / df[genre].sum()
        for genre in genre_columns
    }
    average_ratings_series = pd.Series(average_ratings_by_genre).sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    average_ratings_series.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Average IMDb Ratings by Genre', fontsize=16)
    plt.xlabel('Genres', fontsize=14)
    plt.ylabel('Average IMDb Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    chart_path = os.path.join(static_dir, 'AverageIMDbRatingsByGenre.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# Create the chart when the server starts
chart_path = create_genre_ratings_chart(df)

# Load the pre-trained model
model = joblib.load('movie_genre_model.pkl')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data (budget and genre)
        budget = request.form['budget']
        genre = request.form['genre']

        # Prepare the input dataframe
        input_data = {
            'budget': [budget],
            'genre': [genre],  # Genre can be expanded if needed
        }

        # Convert to DataFrame
        df_input = pd.DataFrame(input_data)

        # Clean the data (same as in training)
        df_input = clean_data(df_input)

        # One-hot encode genres (same as in training)
        df_input = one_hot_encode_genres(df_input)

        # Ensure all columns are present (add missing ones with default values)
        model_columns = [col for col in df.columns if col != 'avg_vote']  # Exclude 'avg_vote' from model_columns
        missing_columns = set(model_columns) - set(df_input.columns)

        # Add missing columns with default values
        for col in missing_columns:
            if col == 'avg_vote':
                df_input[col] = df['avg_vote'].median()  # Set to median if missing
            else:
                df_input[col] = 0  # Set missing genre columns to 0

        # Reorder columns to match the model training order
        df_input = df_input[model_columns]

        # Predict using the model
        prediction = model.predict(df_input)

        # Return the prediction to the user
        return render_template('home.html', chart_url=url_for('static', filename='AverageIMDbRatingsByGenre.png'), prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form input from the request
            budget = request.form['budget']
            genre = request.form['genre']
            
            # Prepare data for prediction (you may need more preprocessing here)
            input_data = {
                'budget': budget,  # make sure you include other features if necessary
                'genre': genre,
                # include other fields as needed by your model
            }

            # Send a request to the prediction route
            prediction_response = request.post('/predict', json=input_data)

            if prediction_response.status_code == 200:
                prediction = prediction_response.json().get('prediction')

        except Exception as e:
            print(f"Error: {e}")

    chart_url = url_for('static', filename='AverageIMDbRatingsByGenre.png')
    return render_template('home.html', chart_url=chart_url, prediction=prediction)



# Route to serve the chart
@app.route('/chart')
def chart():
    return send_file(chart_path, mimetype='image/png')

# Endpoint to get the entire dataset
@app.route('/data', methods=['GET'])
def get_data():
    data = df.to_dict(orient='records')  # Convert rows to list of dictionaries
    return jsonify(data)

# Endpoint to get a sample of the data (first 5 rows)
@app.route('/data/sample', methods=['GET'])
def get_sample_data():
    sample_data = df.head().to_dict(orient='records')
    return jsonify(sample_data)

if __name__ == '__main__':
    # Save the cleaned dataset if needed
    cleaned_file_path = 'cleaned_data.csv'
    df.to_csv(cleaned_file_path, index=False)
    
    # Start the Flask app
    app.run(debug=True)
