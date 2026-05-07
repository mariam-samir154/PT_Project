import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import nltk

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)


st.set_page_config(
    page_title="Movie Popularity Prediction",
    layout="wide"
)

st.title("🎬 Movie Popularity Prediction System")
st.write("Upload a CSV file and evaluate all trained models.")


@st.cache_resource
def load_models():

    objects = {
        "imputer": pickle.load(open("imputer.pkl", "rb")),
        "model_runtime": pickle.load(open("model_runtime.pkl", "rb")),
        "mlb_countries": pickle.load(open("mlb_countries.pkl", "rb")),
        "mlb_genres": pickle.load(open("mlb_genres.pkl", "rb")),
        "scaler": pickle.load(open("scaler.pkl", "rb")),
        "top_langs": pickle.load(open("top_langs.pkl", "rb")),
        "top_companies": pickle.load(open("top_companies.pkl", "rb")),
        "top5_countries": pickle.load(open("top5_countries.pkl", "rb")),
        "top_features_clean": pickle.load(open("top_features_clean.pkl", "rb")),
        "train_age_median": pickle.load(open("train_age_median.pkl", "rb")),
        "roi_median": pickle.load(open("roi_median.pkl", "rb")),

        "ridge_model": pickle.load(open("model_ridge.pkl", "rb")),
        "lasso_model": pickle.load(open("model_lasso.pkl", "rb")),
        "rf_model": pickle.load(open("model_rf.pkl", "rb")),
        "lgb_model": pickle.load(open("model_lgb1000.pkl", "rb")),
        "xgb_model": pickle.load(open("model_XGB.pkl", "rb")),
    }

    return objects


models = load_models()


uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


def safe_col(df, col, default=""):
    if col not in df.columns:
        df[col] = default
    return df[col]


def preprocess(df):

    top_langs = models["top_langs"]
    top_companies = models["top_companies"]
    top5_countries = models["top5_countries"]

    
    required_columns = [
        'overview',
        'production_companies',
        'production_countries',
        'genres',
        'status',
        'quality',
        'original_language',
        'release_date',
        'budget',
        'revenue',
        'runtime',
        'vote_average',
        'vote_count',
        'theatrical',
        'popularity'
    ]

    for col in required_columns:
        safe_col(df, col)


    y_test = np.log1p(df['popularity'].fillna(0))

   
    df['overview'] = df['overview'].fillna('').astype(str)

    no_overview_mask = df['overview'].str.strip() == ''

    df['overview_len'] = df['overview'].str.len()
    df['word_count'] = df['overview'].str.split().str.len()

    sid = SentimentIntensityAnalyzer()

    scores = df['overview'].apply(
        lambda x: sid.polarity_scores(str(x))
    )

    df['sentiment_pos'] = scores.apply(lambda x: x['pos'])
    df['sentiment_neg'] = scores.apply(lambda x: x['neg'])
    df['sentiment_compound'] = scores.apply(lambda x: x['compound'])

  
    df['original_language'] = df['original_language'].apply(
        lambda x: x if x in top_langs else 'other'
    )

    df = pd.get_dummies(
        df,
        columns=['original_language'],
        prefix='lang'
    )

    for lang_col in [f'lang_{l}' for l in list(top_langs) + ['other']]:
        if lang_col not in df.columns:
            df[lang_col] = 0

    
    df['is_released'] = (df['status'] == 'Released').astype(int)

    
    df['release_date'] = pd.to_datetime(
        df['release_date'],
        errors='coerce'
    )

    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month

    
    df['vote_count_log'] = np.log1p(
        pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    )

    
    df['production_companies_list'] = (
        df['production_companies']
        .fillna('')
        .apply(lambda x: [c.strip() for c in str(x).split(',')] if x != '' else [])
    )

    df['num_production_companies'] = (
        df['production_companies_list'].apply(len)
    )

    df['no_of_large_production_companies'] = (
        df['production_companies_list']
        .apply(lambda companies: int(sum(c in top_companies for c in companies)))
    )

    
    df['production_countries_list'] = (
        df['production_countries']
        .fillna('')
        .apply(lambda x: [c.strip() for c in str(x).split(',')] if x != '' else [])
    )

    def map_countries(countries):
        mapped = [c if c in top5_countries else 'other' for c in countries]
        return list(set(mapped)) or ['other']

    df['production_countries_mapped'] = (
        df['production_countries_list']
        .apply(map_countries)
    )

    countries_matrix = models["mlb_countries"].transform(
        df['production_countries_mapped']
    )

    countries_df = pd.DataFrame(
        countries_matrix,
        columns=models["mlb_countries"].classes_,
        index=df.index
    )

    df = pd.concat([df, countries_df], axis=1)

    
    genre_groups = {
        'action_group': ['Action','Adventure','Thriller','War'],
        'comedy_group': ['Comedy','Family'],
        'drama_group': ['Drama','History'],
        'romance_group': ['Romance'],
        'sci_fi_group': ['Science Fiction','Fantasy'],
        'dark_group': ['Horror','Crime','Mystery']
    }

    df['all_genres'] = df['genres'].apply(
        lambda x: x.split(', ') if isinstance(x, str) else []
    )

    def map_to_groups(genres):
        groups = []

        for gname, glist in genre_groups.items():
            if any(g in genres for g in glist):
                groups.append(gname)

        return groups

    df['genre_groups_list'] = df['all_genres'].apply(map_to_groups)

    genre_matrix = models["mlb_genres"].transform(
        df['genre_groups_list']
    )

    genre_df = pd.DataFrame(
        genre_matrix,
        columns=models["mlb_genres"].classes_,
        index=df.index
    )

    df = pd.concat([df, genre_df], axis=1)

    
    numeric_cols = [
        'budget',
        'revenue',
        'runtime',
        'vote_average',
        'theatrical'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['budget_log'] = np.log1p(df['budget'].fillna(0))
    df['revenue_log'] = np.log1p(df['revenue'].fillna(0))

    
    X = df.copy()

    drop_cols = [
        'popularity',
        'overview',
        'genres',
        'production_companies',
        'production_countries',
        'release_date',
        'all_genres',
        'genre_groups_list',
        'production_companies_list',
        'production_countries_list',
        'production_countries_mapped'
    ]

    X.drop(columns=drop_cols, errors='ignore', inplace=True)

    X = X.fillna(0)

    for col in models["top_features_clean"]:
        if col not in X.columns:
            X[col] = 0

    X_final = X[models["top_features_clean"]]

    X_scaled = models["scaler"].transform(X_final)

    return X_final, X_scaled, y_test


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        X_final, X_scaled, y_test = preprocess(df)

        models_dict = {
            'Ridge Regression': (
                models["ridge_model"],
                X_scaled
            ),

            'Lasso Regression': (
                models["lasso_model"],
                X_scaled
            ),

            'Random Forest': (
                models["rf_model"],
                X_final
            ),

            'LightGBM': (
                models["lgb_model"],
                X_final
            ),

            'XGBoost': (
                models["xgb_model"],
                X_final
            )
        }

        results = []

        for name, (model, X_input) in models_dict.items():

            preds = model.predict(X_input)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            results.append({
                "Model": name,
                "MSE": round(mse, 4),
                "R2 Score": round(r2, 4)
            })

        results_df = pd.DataFrame(results)

        st.subheader("📊 Model Results")
        st.dataframe(results_df)

        
        best_model = results_df.sort_values(
            by='R2 Score',
            ascending=False
        ).iloc[0]

        st.success(
            f"🏆 Best Model: {best_model['Model']} "
            f"(R² = {best_model['R2 Score']})"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")