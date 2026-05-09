import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import nltk

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.impute import IterativeImputer


warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)


st.set_page_config(
    page_title="Movie Popularity Prediction",
    layout="wide"
)

st.title("🎬 Movie Popularity Prediction System")
st.write("Upload a CSV file and evaluate all trained models.")


st.markdown("---")
st.header("🏷️ Regression ")

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

        st.subheader("📊 Regression Results")
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


# ══════════════════════════════════════════════════════
#  MILESTONE 2 — CLASSIFICATION (append after their code)
# ══════════════════════════════════════════════════════

st.markdown("---")
st.header("🏷️ Classification ")

cls_uploaded_file = st.file_uploader(
    "Upload Classification CSV File",
    type=["csv"],
    key="cls_upload"
)

@st.cache_data
def load_cls_models():
    return {
        "imputer":           pickle.load(open("cls_imputer.pkl",           "rb")),
        "model_runtime":     pickle.load(open("cls_model_runtime.pkl",     "rb")),
        "mlb_countries":     pickle.load(open("cls_mlb_countries.pkl",     "rb")),
        "mlb_genres":        pickle.load(open("cls_mlb_genres.pkl",        "rb")),
        "scaler":            pickle.load(open("cls_scaler.pkl",            "rb")),
        "top_langs":         pickle.load(open("cls_top_langs.pkl",         "rb")),
        "top_companies":     pickle.load(open("cls_top_companies.pkl",     "rb")),
        "top5_countries":    pickle.load(open("cls_top5_countries.pkl",    "rb")),
        "selected_features": pickle.load(open("cls_selected_features.pkl", "rb")),
        "train_age_median":  pickle.load(open("cls_train_age_median.pkl",  "rb")),
        "roi_median":        pickle.load(open("cls_roi_median.pkl",        "rb")),
        "label_map":         pickle.load(open("cls_label_map.pkl",         "rb")),
        "lr_model":          pickle.load(open("cls_model_lr.pkl",          "rb")),
        "rf_model":          pickle.load(open("cls_model_rf.pkl",          "rb")),
        "xgb_model":         pickle.load(open("cls_model_xgb.pkl",        "rb")),
        "lgbm_model":        pickle.load(open("cls_model_lgbm.pkl",       "rb")),
        "cat_model":         pickle.load(open("cls_model_cat.pkl",        "rb")),
        "et_model":          pickle.load(open("cls_model_et.pkl",         "rb")),
    }

cls_models = load_cls_models()

def preprocess_classification(df):
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    top_langs      = cls_models["top_langs"]
    top_companies  = cls_models["top_companies"]
    top5_countries = cls_models["top5_countries"]

    label_map = cls_models["label_map"]
    # normalize mapping keys



     # reverse if saved backwards
    if isinstance(list(label_map.keys())[0], int):
      label_map = {v: k for k, v in label_map.items()}


    label_map = {
        str(k).strip().lower(): v
        for k, v in label_map.items()
    }

    if 'popularityLevel' in df.columns:

        df['popularityLevel'] = (
           df['popularityLevel']
           .astype(str)
           .str.strip()
           .str.lower()
        )

        y_test = df['popularityLevel'].map(label_map)
    
        if y_test.isna().any():

            bad_labels = (
              df.loc[y_test.isna(), 'popularityLevel']
              .unique()
               .tolist()
            )

            raise ValueError(
                f"Unknown popularityLevel labels found: {bad_labels}"
            )


       
    elif 'target' in df.columns:
        y_test = df['target']
    else:
        raise ValueError("CSV must contain a 'popularityLevel' or 'target' column.")

    for col in ['overview', 'production_companies', 'production_countries',
                'genres', 'status', 'quality', 'original_language',
                'release_date', 'budget', 'revenue', 'runtime',
                'vote_average', 'vote_count', 'theatrical']:
        if col not in df.columns:
            df[col] = np.nan

    # Overview text features
    df['overview'] = df['overview'].fillna('').astype(str)
    no_overview_mask = df['overview'].str.strip() == ''
    df['has_overview']    = (~no_overview_mask).astype(int)
    df['overview_len']    = df['overview'].str.len()
    df['word_count']      = df['overview'].str.split().str.len()
    df['has_based_on']    = df['overview'].str.contains(r'based on',   case=False, na=False).astype(int)
    df['has_true_story']  = df['overview'].str.contains(r'true story', case=False, na=False).astype(int)
    df['has_sequel_hint'] = df['overview'].str.contains(r'\bsequel\b|\bpart \d\b|\bchapter \d\b', case=False, na=False).astype(int)
    df['theme_action']    = df['overview'].str.contains(r'war|battle|fight|mission|agent|weapon', case=False, na=False).astype(int)
    df['theme_romance']   = df['overview'].str.contains(r'love|romance|relationship|heart',       case=False, na=False).astype(int)
    df['theme_horror']    = df['overview'].str.contains(r'terror|haunted|demon|evil|supernatural|fear', case=False, na=False).astype(int)
    df['theme_family']    = df['overview'].str.contains(r'family|daughter|son|father|mother|child', case=False, na=False).astype(int)
    df['theme_scifi']     = df['overview'].str.contains(r'alien|space|future|robot|galaxy|planet',  case=False, na=False).astype(int)

    sid = SentimentIntensityAnalyzer()
    scores = df['overview'].apply(lambda x: sid.polarity_scores(str(x)))
    df['sentiment_pos']      = scores.apply(lambda x: x['pos'])
    df['sentiment_neg']      = scores.apply(lambda x: x['neg'])
    df['sentiment_compound'] = scores.apply(lambda x: x['compound'])
    df.drop(columns=['overview'], inplace=True)

    for col in ['title', 'original_title', 'adult', 'homepage', 'imdb_id',
                'backdrop_path', 'poster_path', 'tagline', 'text_combined', 'id']:
        df.drop(columns=[col], errors='ignore', inplace=True)

    df['theatrical']     = pd.to_numeric(df['theatrical'], errors='coerce').fillna(0).astype(int)
    df['vote_count_log'] = np.log1p(pd.to_numeric(df['vote_count'], errors='coerce').fillna(0))
    df.drop(columns=['vote_count'], inplace=True)

    df['original_language'] = df['original_language'].apply(lambda x: x if x in top_langs else 'other')
    df = pd.get_dummies(df, columns=['original_language'], prefix='lang')
    for lang_col in [f'lang_{l}' for l in list(top_langs) + ['other']]:
        if lang_col not in df.columns:
            df[lang_col] = 0

    df['is_released']    = (df['status'] == 'Released').astype(int)
    df['is_in_pipeline'] = df['status'].isin(['In Production','Post Production','Planned']).astype(int)
    df['is_uncertain']   = (df['status'] == 'Rumored').astype(int)
    df['is_canceled']    = (df['status'] == 'Canceled').astype(int)
    df.drop(columns=['status'], inplace=True)

    df['production_companies_list'] = df['production_companies'].fillna('').apply(
        lambda x: [c.strip() for c in str(x).split(',')] if x != '' else [])
    df['no_of_large_production_companies'] = df['production_companies_list'].apply(
        lambda cs: int(sum(c in top_companies for c in cs)))
    df['num_production_companies'] = df['production_companies_list'].apply(len)
    df.drop(columns=['production_companies', 'production_companies_list'], inplace=True)

    quality_map    = {'spam': -1, 'stub': 0, 'real': 1}
    confidence_map = {'confident': 3, 'legitimate': 2, 'likely': 1, 'uncertain': 0}
    df['quality_type']       = df['quality'].str.split('_').str[0].map(quality_map).fillna(0)
    df['quality_confidence'] = df['quality'].str.split('_').str[1].map(confidence_map).fillna(0)
    df.drop(columns=['quality', 'quality_type'], inplace=True)

    X = df.drop(columns=['popularity', 'log_popularity', 'row_null_pct',
                          'popularityLevel', 'target'], errors='ignore')

    X['release_date']         = pd.to_datetime(X['release_date'], errors='coerce')
    X['release_date_missing'] = X['release_date'].isna().astype(int)
    X['release_year']         = X['release_date'].dt.year
    X['release_month']        = X['release_date'].dt.month
    X.drop(columns=['release_date'], inplace=True)
    X['budget']  = pd.to_numeric(X['budget'],  errors='coerce').replace(0, np.nan)
    X['revenue'] = pd.to_numeric(X['revenue'], errors='coerce').replace(0, np.nan)

    mice_features = ['release_year','release_month','revenue','budget',
                     'vote_average','vote_count_log','runtime','theatrical']
    released_mask = X['is_released'] == 1
    X.loc[released_mask, mice_features] = cls_models["imputer"].transform(X.loc[released_mask, mice_features])
    X['release_year']  = X['release_year'].round().clip(1880, 2026)
    X['release_month'] = X['release_month'].round().clip(1, 12)
    not_released_mask  = X['is_released'] == 0
    X.loc[not_released_mask & X['release_year'].isna(),  'release_year']  = -1
    X.loc[not_released_mask & X['release_month'].isna(), 'release_month'] = -1

    def get_season(m):
        if m in [12,1,2]: return 1
        elif m in [3,4,5]: return 2
        elif m in [6,7,8]: return 3
        elif m in [9,10,11]: return 4
        else: return -1

    X['release_season'] = X['release_month'].apply(get_season)
    X['release_decade'] = X.apply(
        lambda row: (int(row['release_year']//10)*10) if row['release_date_missing']==0 else np.nan, axis=1)
    X['decade_unknown'] = X['release_decade'].isna().astype(int)
    X['release_decade'] = X['release_decade'].fillna(-1)
    X['movie_age'] = np.where(X['release_date_missing']==0, 2026 - X['release_year'], np.nan)
    X['movie_age'] = X['movie_age'].fillna(cls_models["train_age_median"])

    X['runtime'] = pd.to_numeric(X['runtime'], errors='coerce')
    X['runtime'] = X['runtime'].where((X['runtime'] > 0) & (X['runtime'] <= 400), np.nan)
    features_rt  = ['vote_count_log','quality_confidence','vote_average','is_released']
    unknown_rt   = X[X['runtime'].isna()]
    if len(unknown_rt) > 0:
        X.loc[X['runtime'].isna(), 'runtime'] = cls_models["model_runtime"].predict(unknown_rt[features_rt])

    X['production_countries_list'] = X['production_countries'].fillna('').apply(
        lambda x: [c.strip() for c in str(x).split(',')] if x != '' else [])
    def map_countries(countries):
        mapped = [c if c in top5_countries else 'other' for c in countries]
        return list(set(mapped)) or ['other']
    X['production_countries_mapped'] = X['production_countries_list'].apply(map_countries)
    countries_matrix = cls_models["mlb_countries"].transform(X['production_countries_mapped'])
    countries_df = pd.DataFrame(countries_matrix, columns=cls_models["mlb_countries"].classes_, index=X.index)
    X = pd.concat([X, countries_df], axis=1)
    X.drop(columns=['production_countries','production_countries_list','production_countries_mapped'], inplace=True)

    genre_groups = {
        'action_group': ['Action','Adventure','Thriller','War'],
        'comedy_group': ['Comedy','Family'],
        'drama_group':  ['Drama','History'],
        'romance_group':['Romance'],
        'sci_fi_group': ['Science Fiction','Fantasy'],
        'dark_group':   ['Horror','Crime','Mystery'],
        'other_group':  ['Documentary','Music','TV Movie','Western','Animation']
    }
    X['all_genres'] = X['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    X['has_genres'] = X['genres'].notna().astype(int)
    X['num_genres'] = X['all_genres'].apply(len)

    def map_to_groups(genres):
        if not isinstance(genres, list): genres = []
        return [gn for gn, gl in genre_groups.items() if any(g in genres for g in gl)]

    def fill_missing_groups(row):
        if row['has_genres']: return row['genre_groups_list']
        inferred = []
        if row.get('theme_action'):  inferred.append('action_group')
        if row.get('theme_romance'): inferred.append('romance_group')
        if row.get('theme_scifi'):   inferred.append('sci_fi_group')
        if row.get('theme_horror'):  inferred.append('dark_group')
        if row.get('theme_family'):  inferred.append('comedy_group')
        return inferred

    X['genre_groups_list'] = X['all_genres'].apply(map_to_groups)
    missing_genre_mask = X['genres'].isna()
    X.loc[missing_genre_mask, 'genre_groups_list'] = \
        X.loc[missing_genre_mask].apply(fill_missing_groups, axis=1)
    group_matrix = cls_models["mlb_genres"].transform(X['genre_groups_list'])
    group_df = pd.DataFrame(group_matrix, columns=cls_models["mlb_genres"].classes_, index=X.index)
    X = pd.concat([X, group_df], axis=1)
    X.drop(columns=['genres','spoken_languages','all_genres','genre_groups_list'],
           errors='ignore', inplace=True)

    X['budget_known']   = (X['budget'] > 0).astype(int)
    X['revenue_known']  = (X['revenue'] > 0).astype(int)
    X['both_known']     = ((X['budget'] > 0) & (X['revenue'] > 0)).astype(int)
    X['is_blockbuster'] = ((X['budget'] >= 20_000_000) & (X['revenue'] >= 50_000_000)).astype(int)
    X['budget_log']     = np.log1p(X['budget'].fillna(0))
    X['revenue_log']    = np.log1p(X['revenue'].fillna(0))
    roi_mask = (X['budget'] > 0) & (X['revenue'] > 0)
    X['roi'] = np.where(roi_mask, (X['revenue'] - X['budget']) / X['budget'], np.nan)
    X['roi'] = X['roi'].fillna(cls_models["roi_median"])
    X.drop(columns=['budget','revenue'], inplace=True)

    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    X = X.fillna(0)

    for col in cls_models["selected_features"]:
        if col not in X.columns:
            X[col] = 0
    X_final  = X[cls_models["selected_features"]]
    X_scaled = cls_models["scaler"].transform(X_final)

    return X_final, X_scaled, y_test


if cls_uploaded_file is not None:
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        df_cls = pd.read_csv(cls_uploaded_file)
        st.success("Classification CSV uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df_cls.head())

        X_final_c, X_scaled_c, y_test_c = preprocess_classification(df_cls)

        CLASS_NAMES = ['Very Low', 'Low', 'Medium', 'High']

        cls_dict = {
            'Logistic Regression (SGD)': cls_models["lr_model"],
            'Random Forest':             cls_models["rf_model"],
            'XGBoost':                   cls_models["xgb_model"],
            'LightGBM':                  cls_models["lgbm_model"],
            'CatBoost':                  cls_models["cat_model"],
            'Extra Trees':               cls_models["et_model"],
        }

        cls_results = []
        for name, model in cls_dict.items():
            preds = model.predict(X_scaled_c)
            cls_results.append({
                "Model":    name,
                "Accuracy": round(accuracy_score(y_test_c, preds), 4),
                "Macro F1": round(f1_score(y_test_c, preds, average='macro'), 4),
            })

        cls_results_df = pd.DataFrame(cls_results)

        st.subheader("📊 Classification Results")
        st.dataframe(cls_results_df)

        best_cls = cls_results_df.sort_values("Macro F1", ascending=False).iloc[0]
        st.success(
            f"🏆 Best Model: {best_cls['Model']} "
            f"(Accuracy = {best_cls['Accuracy']}, Macro F1 = {best_cls['Macro F1']})"
        )

        st.subheader(f"📋 Detailed Report — {best_cls['Model']}")
        best_preds_c = cls_dict[best_cls['Model']].predict(X_scaled_c)
        st.code(classification_report(y_test_c, best_preds_c,
                                       target_names=CLASS_NAMES, zero_division=0))

    except Exception as e:
        st.error(f"Error: {str(e)}")