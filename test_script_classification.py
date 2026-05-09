import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)


# ── Load test data ─────────────────────────────────────────────────────────────
test_path = 'test_data_classification.csv'
df = pd.read_csv(test_path)


# ── Load preprocessing artefacts ──────────────────────────────────────────────
imputer            = pickle.load(open('cls_imputer.pkl',            'rb'))
model_runtime      = pickle.load(open('cls_model_runtime.pkl',      'rb'))
mlb_countries      = pickle.load(open('cls_mlb_countries.pkl',      'rb'))
mlb_genres         = pickle.load(open('cls_mlb_genres.pkl',         'rb'))
scaler             = pickle.load(open('cls_scaler.pkl',             'rb'))
top_langs          = pickle.load(open('cls_top_langs.pkl',          'rb'))
top_companies      = pickle.load(open('cls_top_companies.pkl',      'rb'))
top5_countries     = pickle.load(open('cls_top5_countries.pkl',     'rb'))
SELECTED_FEATURES  = pickle.load(open('cls_selected_features.pkl',  'rb'))
train_age_median   = pickle.load(open('cls_train_age_median.pkl',   'rb'))
roi_median         = pickle.load(open('cls_roi_median.pkl',         'rb'))
#revenue_log_median = pickle.load(open('cls_revenue_log_median.pkl', 'rb'))
label_map          = pickle.load(open('cls_label_map.pkl',          'rb'))

# ── Load all trained models ────────────────────────────────────────────────────
# ADASYN (primary)
lr_model    = pickle.load(open('cls_model_lr.pkl',    'rb'))
rf_model    = pickle.load(open('cls_model_rf.pkl',    'rb'))
xgb_model   = pickle.load(open('cls_model_xgb.pkl',  'rb'))
lgbm_model  = pickle.load(open('cls_model_lgbm.pkl', 'rb'))
cat_model   = pickle.load(open('cls_model_cat.pkl',  'rb'))
et_model    = pickle.load(open('cls_model_et.pkl',   'rb'))

# SMOTE (experiment 2)
lr_smote    = pickle.load(open('cls_model_lr_smote.pkl',   'rb'))
lgbm_smote  = pickle.load(open('cls_model_lgbm_smote.pkl', 'rb'))
xgb_smote   = pickle.load(open('cls_model_xgb_smote.pkl',  'rb'))

# No-synthetic (experiment 3)
lr_nosyn    = pickle.load(open('cls_model_lr_nosyn.pkl',   'rb'))
lgbm_nosyn  = pickle.load(open('cls_model_lgbm_nosyn.pkl', 'rb'))
xgb_nosyn   = pickle.load(open('cls_model_xgb_nosyn.pkl',  'rb'))

# Best overall model
best_model      = pickle.load(open('cls_best_model.pkl',      'rb'))
best_model_name = pickle.load(open('cls_best_model_name.pkl', 'rb'))


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (no row dropping — requirement compliant)
# ══════════════════════════════════════════════════════════════════════════════

# 1. row_null_pct  (computed but NOT used to drop anything)
no_null_cols = ['id', 'quality', 'theatrical', 'runtime', 'status', 'revenue',
                'vote_average', 'vote_count', 'budget', 'original_language', 'popularityLevel']
df_nulls = df.drop(columns=[c for c in no_null_cols if c in df.columns], errors='ignore')
df['row_null_pct'] = df_nulls.isnull().mean(axis=1)   # kept as feature, no rows dropped

# 2. Drop emotion/VAD columns
emotion_cols = ['movie_intensity_anger','movie_intensity_disgust','movie_intensity_joy',
                'movie_intensity_sadness','movie_intensity_trust','movie_intensity_fear',
                'movie_intensity_surprise','movie_intensity_anticipation',
                'movie_vad_valence','movie_vad_arousal','movie_vad_dominance',
                'movie_valence','movie_scl_shift','movie_scl_coverage']
df.drop(columns=[c for c in emotion_cols if c in df.columns], inplace=True)

# 3. Target — encode but keep original for evaluation
if 'popularityLevel' in df.columns:
    y_test = df['popularityLevel'].map(label_map)
else:
    raise ValueError("Test CSV must have a 'popularityLevel' column.")

# 4. Drop id, target columns
df.drop(columns=['id', 'popularityLevel'], errors='ignore', inplace=True)

# 5. text_null_count + meta_null_count  (missingness features — no saving needed)
#    These are derived purely from columns that already exist in the row,
#    so they are recomputable on any new sample without fitting anything.
df['overview'] = df['overview'].fillna('').astype(str)
df['overview'] = df['overview'].replace('', None).fillna('no overview')
no_overview_mask = df['overview'].str.strip().str.lower() == 'no overview'
df['has_overview'] = (~no_overview_mask).astype(int)

df['text_null_count'] = (1 - df['has_overview'])   # genres_was_null always 0 on test

null_flag_cols = [c for c in df.columns if '_was_null' in c or '_missing' in c]
meta_flags = [c for c in null_flag_cols if any(x in c for x in ['production','spoken','language'])]
df['meta_null_count'] = df[meta_flags].sum(axis=1) if meta_flags else 0

# 6. Overview text features
df['overview_len']    = df['overview'].str.len().fillna(0)
df['word_count']      = df['overview'].str.split().str.len().fillna(0)
df['has_based_on']    = df['overview'].str.contains(r'based on',   case=False, na=False).astype(int)
df['has_true_story']  = df['overview'].str.contains(r'true story', case=False, na=False).astype(int)
df['has_sequel_hint'] = df['overview'].str.contains(r'\bsequel\b|\bpart \d\b|\bchapter \d\b', case=False, na=False).astype(int)
df['theme_action']    = df['overview'].str.contains(r'war|battle|fight|mission|agent|weapon', case=False, na=False).astype(int)
df['theme_romance']   = df['overview'].str.contains(r'love|romance|relationship|heart',       case=False, na=False).astype(int)
df['theme_horror']    = df['overview'].str.contains(r'terror|haunted|demon|evil|supernatural|fear', case=False, na=False).astype(int)
df['theme_family']    = df['overview'].str.contains(r'family|daughter|son|father|mother|child', case=False, na=False).astype(int)
df['theme_scifi']     = df['overview'].str.contains(r'alien|space|future|robot|galaxy|planet',  case=False, na=False).astype(int)

df.loc[no_overview_mask, ['overview_len','word_count','has_based_on','has_true_story',
                           'has_sequel_hint','theme_action','theme_romance','theme_horror',
                           'theme_family','theme_scifi']] = 0

sid = SentimentIntensityAnalyzer()
df.loc[no_overview_mask, ['sentiment_pos','sentiment_neg','sentiment_compound']] = 0
scores = df.loc[~no_overview_mask, 'overview'].apply(lambda x: sid.polarity_scores(x))
df.loc[~no_overview_mask, 'sentiment_pos']      = scores.apply(lambda x: x['pos'])
df.loc[~no_overview_mask, 'sentiment_neg']      = scores.apply(lambda x: x['neg'])
df.loc[~no_overview_mask, 'sentiment_compound'] = scores.apply(lambda x: x['compound'])
df[['sentiment_pos','sentiment_neg','sentiment_compound']] = \
    df[['sentiment_pos','sentiment_neg','sentiment_compound']].fillna(0)
df.drop(columns=['overview'], inplace=True)

# 7. Drop useless columns
for col in ['title','original_title','adult','homepage','imdb_id',
            'backdrop_path','poster_path','tagline','text_combined','avg_word_length']:
    df.drop(columns=[col], errors='ignore', inplace=True)

# 8. Theatrical
df['theatrical'] = pd.to_numeric(df['theatrical'], errors='coerce').fillna(0).astype(int)

# 9. Vote count log
df['vote_count_log'] = np.log1p(pd.to_numeric(df['vote_count'], errors='coerce').fillna(0))
df.drop(columns=['vote_count'], inplace=True)

# 10. Original language
df['original_language'] = df['original_language'].apply(lambda x: x if x in top_langs else 'other')
df = pd.get_dummies(df, columns=['original_language'], prefix='lang')
for lang_col in [f'lang_{l}' for l in list(top_langs) + ['other']]:
    if lang_col not in df.columns:
        df[lang_col] = 0

# 11. Status flags
df['is_released']    = (df['status'] == 'Released').astype(int)
df['is_in_pipeline'] = df['status'].isin(['In Production','Post Production','Planned']).astype(int)
df['is_uncertain']   = (df['status'] == 'Rumored').astype(int)
df['is_canceled']    = (df['status'] == 'Canceled').astype(int)
df.drop(columns=['status'], inplace=True)

# 12. Production companies
df['production_companies_list'] = df['production_companies'].fillna('').apply(
    lambda x: [c.strip() for c in x.split(',')] if x != '' else [])
df['no_of_large_production_companies'] = df['production_companies_list'].apply(
    lambda cs: int(sum(c in top_companies for c in cs)))
df['num_production_companies'] = df['production_companies_list'].apply(len)
df.drop(columns=['production_companies','production_companies_list'], inplace=True)

# 13. Quality
quality_map    = {'spam': -1, 'stub': 0, 'real': 1}
confidence_map = {'confident': 3, 'legitimate': 2, 'likely': 1, 'uncertain': 0}
df['quality_type']       = df['quality'].str.split('_').str[0].map(quality_map).fillna(0)
df['quality_confidence'] = df['quality'].str.split('_').str[1].map(confidence_map).fillna(0)
df.drop(columns=['quality','quality_type'], inplace=True)

# 14. Build X
X = df.drop(columns=['popularity','log_popularity','row_null_pct',
                      'target','popularityLevel'], errors='ignore')

# 15. Release date → temporal features
X['release_date']         = pd.to_datetime(X['release_date'], errors='coerce')
X['release_date_missing'] = X['release_date'].isna().astype(int)
X['release_year']         = X['release_date'].dt.year
X['release_month']        = X['release_date'].dt.month
X.drop(columns=['release_date'], inplace=True)
X['budget']  = pd.to_numeric(X['budget'],  errors='coerce').replace(0, np.nan)
X['revenue'] = pd.to_numeric(X['revenue'], errors='coerce').replace(0, np.nan)

# 16. MICE imputation (released rows only)
mice_features = ['release_year','release_month','revenue','budget',
                 'vote_average','vote_count_log','runtime','theatrical']
released_mask = X['is_released'] == 1
X.loc[released_mask, mice_features] = imputer.transform(X.loc[released_mask, mice_features])
X['release_year']  = X['release_year'].round().clip(1880, 2026)
X['release_month'] = X['release_month'].round().clip(1, 12)

not_released_mask = X['is_released'] == 0
X.loc[not_released_mask & X['release_year'].isna(),  'release_year']  = -1
X.loc[not_released_mask & X['release_month'].isna(), 'release_month'] = -1

def get_season(m):
    if m in [12,1,2]:    return 1
    elif m in [3,4,5]:   return 2
    elif m in [6,7,8]:   return 3
    elif m in [9,10,11]: return 4
    else:                 return -1

X['release_season'] = X['release_month'].apply(get_season)
X['release_decade'] = X.apply(
    lambda row: (int(row['release_year']//10)*10) if row['release_date_missing']==0 else np.nan, axis=1)
X['decade_unknown'] = X['release_decade'].isna().astype(int)
X['release_decade'] = X['release_decade'].fillna(-1)
X['movie_age'] = np.where(X['release_date_missing']==0, 2026 - X['release_year'], np.nan)
X['movie_age'] = X['movie_age'].fillna(train_age_median)

# 17. Runtime imputation
X['runtime'] = pd.to_numeric(X['runtime'], errors='coerce')
X['runtime'] = X['runtime'].where((X['runtime'] > 0) & (X['runtime'] <= 400), np.nan)
features_rt  = ['vote_count_log','quality_confidence','vote_average','is_released']
unknown_rt   = X[X['runtime'].isna()]
if len(unknown_rt) > 0:
    X.loc[X['runtime'].isna(), 'runtime'] = model_runtime.predict(unknown_rt[features_rt])

# 18. Production countries
X['production_countries_list'] = X['production_countries'].fillna('').apply(
    lambda x: [c.strip() for c in x.split(',')] if x != '' else [])
def map_countries(countries):
    mapped = [c if c in top5_countries else 'other' for c in countries]
    return list(set(mapped)) or ['other']
X['production_countries_mapped'] = X['production_countries_list'].apply(map_countries)
countries_matrix = mlb_countries.transform(X['production_countries_mapped'])
countries_df = pd.DataFrame(countries_matrix, columns=mlb_countries.classes_, index=X.index)
X = pd.concat([X, countries_df], axis=1)
X.drop(columns=['production_countries','production_countries_list',
                 'production_countries_mapped'], inplace=True)

# 19. Genres
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
group_matrix = mlb_genres.transform(X['genre_groups_list'])
group_df = pd.DataFrame(group_matrix, columns=mlb_genres.classes_, index=X.index)
X = pd.concat([X, group_df], axis=1)
X.drop(columns=['genres','spoken_languages','all_genres','genre_groups_list'],
       errors='ignore', inplace=True)

# 20. Budget / revenue features
X['budget_known']   = (X['budget'] > 0).astype(int)
X['revenue_known']  = (X['revenue'] > 0).astype(int)
X['both_known']     = ((X['budget'] > 0) & (X['revenue'] > 0)).astype(int)
X['is_blockbuster'] = ((X['budget'] >= 20_000_000) & (X['revenue'] >= 50_000_000)).astype(int)
X['budget_log']     = np.log1p(X['budget'].fillna(0))
X['revenue_log']    = np.log1p(X['revenue'].fillna(0))
roi_mask = (X['budget'] > 0) & (X['revenue'] > 0)
X['roi'] = np.where(roi_mask, (X['revenue'] - X['budget']) / X['budget'], np.nan)
X['roi'] = X['roi'].fillna(roi_median)
 # X['revenue_log'] = X['revenue_log'].fillna(revenue_log_median) 
X.drop(columns=['budget','revenue'], inplace=True)

# 21. Bool → int, fill NaN
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)
X = X.fillna(0)

# 22. Align to saved feature list (SELECTED_FEATURES — hardcoded list from training)
for col in SELECTED_FEATURES:
    if col not in X.columns:
        X[col] = 0
X_final  = X[SELECTED_FEATURES]
X_scaled = scaler.transform(X_final)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ['Very Low', 'Low', 'Medium', 'High']

models_to_eval = {
    # ADASYN
    'Logistic Regression [ADASYN]': lr_model,
    'Random Forest [ADASYN]':       rf_model,
    'XGBoost [ADASYN]':             xgb_model,
    'LightGBM [ADASYN]':            lgbm_model,
    'CatBoost [ADASYN]':            cat_model,
    'Extra Trees [ADASYN]':         et_model,
    # SMOTE
    'LR [SMOTE]':                   lr_smote,
    'LightGBM [SMOTE]':             lgbm_smote,
    'XGBoost [SMOTE]':              xgb_smote,
    # No-synthetic
    'LR [No-Synthetic]':            lr_nosyn,
    'LightGBM [No-Synthetic]':      lgbm_nosyn,
    'XGBoost [No-Synthetic]':       xgb_nosyn,
}

print("\n" + "="*60)
print("  CLASSIFICATION RESULTS")
print("="*60)

for name, model in models_to_eval.items():
    preds = model.predict(X_scaled)
    acc   = accuracy_score(y_test, preds)
    macro = f1_score(y_test, preds, average='macro')
    print(f"\n  {name}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Macro F1 : {macro:.4f}")

print("\n" + "="*60)
print(f"  BEST OVERALL MODEL: {best_model_name}")
print("="*60)
best_preds = best_model.predict(X_scaled)
print(f"  Accuracy : {accuracy_score(y_test, best_preds):.4f}")
print(f"  Macro F1 : {f1_score(y_test, best_preds, average='macro'):.4f}")
print(f"\n{classification_report(y_test, best_preds, target_names=CLASS_NAMES)}")
print("="*60)