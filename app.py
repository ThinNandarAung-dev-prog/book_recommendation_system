from flask import Flask, render_template, request
import test3  # This imports your test3.py as a module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendPage')
def recommendPage():
    return render_template('recommendPage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/trending')
def trending():
    return render_template('trending.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']  # Get the input book title from the form

    # Use the get_similar_books function from test3.py
    recommendations = test3.get_similar_books(book_title, top_n=10)

    # Handle the case where no recommendations are found
    if isinstance(recommendations, str):  # If the function returns "Title not found"
        recommendations = []

    # Pass the recommendations back to the HTML template
    return render_template('recommendPage.html', book_title=book_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
