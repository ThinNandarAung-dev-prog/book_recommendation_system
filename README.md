# LitTrio

LitTrio is a small web app that helps you find your next book. Type a title you already like, and it suggests ten others that feel similar — same kind of writing, similar topics, and often the same author or publisher.

This started as a data-mining class project. The site is called **Book Recommender** in the navigation, and the product name on the home page is **LitTrio**.

## What you can do

- **Home** — a short welcome and a way into the rest of the site.
- **Trending** — a gallery of book covers (static images, not live ratings).
- **Recommendation** — search by title and see similar books, with cover, author, and categories.
- **About** — how the project works and who built it.

## How the suggestions are made

The app does not look at what other people bought. It looks at the book itself.

For a sample of 10,000 books it builds a profile from:

- the title
- the categories
- the author
- the publisher
- the publication year

Those profiles are compared with cosine similarity. Books that sit close together in that space are the ones you see as recommendations. Title words count a bit more than the other fields, so a close title match tends to rank higher.

The first time you start the app, this comparison is built in memory. Expect a wait, and give the process a few gigabytes of RAM if you can. After that, searches are just lookups.

## The Amazon dataset is not in this repo

The source file is `amazon/books_data.csv`. It is about 173 MB, which is too large to keep in Git, so `.gitignore` leaves the whole `amazon/` folder out.

Download **books_data.csv** from the public [Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) dataset on Kaggle (you need a free Kaggle account). Put the file here:

```text
amazon/books_data.csv
```

The columns the app expects are `Title`, `authors`, `publisher`, `categories`, `publishedDate`, and `image`. Rows missing any of those are skipped.

`sampled_titles_info.txt` is written automatically when the app loads. It lists the 10,000 titles in the current sample, which is handy when you are not sure which titles will match.

## Run it on your machine

You need Python 3.10 or newer.

```bash
pip install flask pandas scikit-learn numpy
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Search with a title that appears in the sample. Matching ignores extra spaces and capital letters. If the title is not in the 10,000-book sample, the page stays empty and asks you to try another title.

## Project layout

| Path | What it is |
| --- | --- |
| `app.py` | Flask routes for the pages and the search form |
| `test3.py` | Loads the CSV, builds the similarity model, returns similar books |
| `templates/` | HTML pages |
| `static/images/` | Logo, backgrounds, and the trending covers |
| `amazon/books_data.csv` | Local copy of the dataset (not committed) |

## Who made this

- Thin Nandar Aung 

Built with Python, Flask, pandas, scikit-learn, and Tailwind CSS.
