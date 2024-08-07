# Movie Recommendation System

This repository contains a content-based movie recommendation system using Python. The system uses TF-IDF vectorization and cosine similarity to recommend movies based on their features.

## Dataset

The dataset should be a CSV file (`movies.csv`) containing the following columns:

- `index`
- `budget`
- `genres`
- `homepage`
- `id`
- `keywords`
- `original_language`
- `original_title`
- `overview`
- `popularity`
- `production_companies`
- `production_countries`
- `release_date`
- `revenue`
- `runtime`
- `spoken_languages`
- `status`
- `tagline`
- `title`
- `vote_average`
- `vote_count`
- `cast`
- `crew`
- `director`

## Dependencies

Make sure you have the following Python libraries installed:

- `pandas`
- `scikit-learn`

You can install the dependencies using the following command:

```bash
pip install pandas scikit-learn
