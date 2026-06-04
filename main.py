"""
Main Analysis Script — Content-Based Movie Recommendation System
Runs: EDA → Feature Engineering → Similarity Analysis → Recommendation Generation → Report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from recommender import MovieRecommender
import json, os, warnings
warnings.filterwarnings('ignore')

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'font.family': 'DejaVu Sans',
    'axes.titlecolor': '#f0f6fc',
})
ACCENT   = '#58a6ff'
ACCENT2  = '#f78166'
ACCENT3  = '#3fb950'
ACCENT4  = '#d2a8ff'
BG_DARK  = '#0d1117'
BG_MID   = '#161b22'
BG_PANEL = '#21262d'

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CONTENT-BASED MOVIE RECOMMENDATION SYSTEM")
print("  MovieLens Data | Cosine Similarity | TF-IDF Features")
print("=" * 60)

# Step 1: Initialize
print("\n[1/6] Loading data & building recommendation engine...")
rec = MovieRecommender('movies.csv', 'ratings.csv')
stats = rec.get_stats()

print(f"  ✅ Movies loaded     : {stats['total_movies']}")
print(f"  ✅ Ratings loaded    : {stats['total_ratings']}")
print(f"  ✅ Users             : {stats['total_users']}")
print(f"  ✅ Unique genres     : {len(stats['unique_genres'])}")
print(f"  ✅ TF-IDF matrix     : {stats['tfidf_matrix_shape']}")
print(f"  ✅ Cosine sim matrix : {stats['cosine_sim_shape']}")
print(f"  ✅ Year range        : {stats['year_range'][0]}–{stats['year_range'][1]}")
print(f"  ✅ Avg rating        : {stats['avg_rating_overall']}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Generating recommendations...")

# Content-based similar movies
examples = [
    ('Inception (2010)', 8),
    ('The Godfather (1972)', 8),
    ('Get Out (2017)', 8),
    ('Her (2013)', 8),
]
rec_results = {}
for title, n in examples:
    try:
        recs = rec.get_similar_movies(title, top_n=n)
        rec_results[title] = recs
        print(f"\n  Similar to '{title}':")
        for _, row in recs.iterrows():
            print(f"    [{row['similarity_score']:.3f}] {row['title']}  ({row['genres']})")
    except Exception as e:
        print(f"  ⚠️  {e}")

# User-based recommendations
print("\n  User Personalized Recommendations:")
user_recs = {}
for uid in [1, 5, 12, 23]:
    try:
        urec = rec.get_user_recommendations(uid, top_n=5)
        user_recs[uid] = urec
        print(f"\n  User {uid}:")
        for _, row in urec.iterrows():
            print(f"    [{row['recommendation_score']:.3f}] {row['title']}")
    except Exception as e:
        print(f"  ⚠️  User {uid}: {e}")

# Genre-based recommendations
genre_prefs = ['Action', 'Sci-Fi']
genre_recs = rec.get_genre_recommendations(genre_prefs, top_n=8)
print(f"\n  Genre preferences {genre_prefs}:")
for _, row in genre_recs.iterrows():
    print(f"    [{row['genre_match_score']:.3f}] {row['title']}  ({row['genres']})")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Exploratory Data Analysis...")

movies_df = rec.movies_df
ratings_df = rec.ratings_df

# Genre distribution
genre_counts = {}
for g_str in movies_df['genres']:
    for g in g_str.split('|'):
        genre_counts[g.strip()] = genre_counts.get(g.strip(), 0) + 1
genre_series = pd.Series(genre_counts).sort_values(ascending=True)

# Rating distribution
rating_dist = ratings_df['rating'].value_counts().sort_index()

# Movies per year
year_counts = movies_df['year'].dropna().astype(int).value_counts().sort_index()

# Avg rating by genre
genre_ratings = []
for _, row in movies_df.iterrows():
    for g in row['genres'].split('|'):
        genre_ratings.append({'genre': g.strip(), 'avg_rating': row['avg_rating']})
genre_rating_df = pd.DataFrame(genre_ratings)
genre_avg = genre_rating_df.groupby('genre')['avg_rating'].mean().sort_values()

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Creating EDA visualization...")

fig = plt.figure(figsize=(20, 14), facecolor=BG_DARK)
fig.suptitle('Content-Based Movie Recommendation System — EDA Dashboard',
             fontsize=18, color='#f0f6fc', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Genre Distribution ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(genre_series)))
bars = ax1.barh(genre_series.index, genre_series.values, color=colors, edgecolor='none')
ax1.set_title('Genre Distribution', fontsize=13, color='#f0f6fc', pad=10)
ax1.set_xlabel('Number of Movies')
for bar, val in zip(bars, genre_series.values):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=8, color='#8b949e')
ax1.set_facecolor(BG_PANEL)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Panel 2: Stats Card ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(BG_PANEL)
ax2.axis('off')
stats_text = [
    ("📦 Movies",       f"{stats['total_movies']}"),
    ("⭐ Ratings",      f"{stats['total_ratings']:,}"),
    ("👥 Users",        f"{stats['total_users']}"),
    ("🎭 Genres",       f"{len(stats['unique_genres'])}"),
    ("📊 Avg Rating",   f"{stats['avg_rating_overall']}"),
    ("📅 Year Range",   f"{stats['year_range'][0]}–{stats['year_range'][1]}"),
    ("🔢 TF-IDF Dims",  f"{stats['tfidf_matrix_shape'][1]}"),
]
ax2.set_title('Dataset Statistics', fontsize=13, color='#f0f6fc', pad=10)
for i, (label, val) in enumerate(stats_text):
    y = 0.9 - i * 0.13
    ax2.text(0.05, y, label, transform=ax2.transAxes,
             fontsize=10, color='#8b949e', va='top')
    ax2.text(0.95, y, val, transform=ax2.transAxes,
             fontsize=11, color=ACCENT, va='top', ha='right', fontweight='bold')

# ── Panel 3: Rating Distribution ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(rating_dist.index, rating_dist.values,
        color=ACCENT2, edgecolor='none', width=0.4, alpha=0.85)
ax3.set_title('Rating Distribution', fontsize=12, color='#f0f6fc', pad=8)
ax3.set_xlabel('Rating')
ax3.set_ylabel('Count')
ax3.set_facecolor(BG_PANEL)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Panel 4: Movies per Decade ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
decade_counts = (movies_df['year'].dropna()
                 .apply(lambda y: int(y) // 10 * 10)
                 .value_counts().sort_index())
ax4.bar(decade_counts.index, decade_counts.values,
        color=ACCENT3, width=7, edgecolor='none', alpha=0.85)
ax4.set_title('Movies per Decade', fontsize=12, color='#f0f6fc', pad=8)
ax4.set_xlabel('Decade')
ax4.set_ylabel('Count')
ax4.set_facecolor(BG_PANEL)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel 5: Avg Rating by Genre ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
non_zero = genre_avg[genre_avg > 0]
colors5 = plt.cm.Purples(np.linspace(0.4, 0.9, len(non_zero)))
ax5.barh(non_zero.index, non_zero.values, color=colors5, edgecolor='none')
ax5.set_title('Avg Rating by Genre', fontsize=12, color='#f0f6fc', pad=8)
ax5.set_xlabel('Avg Rating')
ax5.axvline(non_zero.mean(), color=ACCENT2, linestyle='--', alpha=0.7, linewidth=1.5)
ax5.set_facecolor(BG_PANEL)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Panel 6: Top Similar Movies to Inception ─────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
inception_recs = rec_results.get('Inception (2010)', pd.DataFrame())
if not inception_recs.empty:
    top8 = inception_recs.head(8)
    short_titles = [t[:35] + '…' if len(t) > 35 else t for t in top8['title']]
    bars6 = ax6.barh(short_titles[::-1], top8['similarity_score'].values[::-1],
                     color=plt.cm.Blues(np.linspace(0.5, 0.9, len(top8))),
                     edgecolor='none')
    ax6.set_title('Top Similar Movies to "Inception (2010)"  [Cosine Similarity]',
                  fontsize=12, color='#f0f6fc', pad=8)
    ax6.set_xlabel('Cosine Similarity Score')
    for bar, val in zip(bars6, top8['similarity_score'].values[::-1]):
        ax6.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8.5, color='#8b949e')
ax6.set_facecolor(BG_PANEL)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# ── Panel 7: Ratings per User ─────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ratings_per_user = ratings_df.groupby('userId').size()
ax7.hist(ratings_per_user, bins=15, color=ACCENT4, edgecolor='none', alpha=0.85)
ax7.set_title('Ratings per User', fontsize=12, color='#f0f6fc', pad=8)
ax7.set_xlabel('# Ratings')
ax7.set_ylabel('Users')
ax7.set_facecolor(BG_PANEL)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

plt.savefig('outputs/eda_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print("  ✅ EDA dashboard saved → outputs/eda_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Creating similarity heatmap...")

fig2, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG_DARK)
fig2.suptitle('Cosine Similarity Analysis', fontsize=16, color='#f0f6fc',
              fontweight='bold', y=1.01)

# Heatmap of first 20 movies
cosine_sim = rec.cosine_sim
labels = [t[:25] + '…' if len(t) > 25 else t
          for t in movies_df['title'].head(20)]

ax_h = axes[0]
im = ax_h.imshow(cosine_sim[:20, :20], cmap='Blues', aspect='auto',
                 vmin=0, vmax=1)
ax_h.set_xticks(range(20))
ax_h.set_yticks(range(20))
ax_h.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
ax_h.set_yticklabels(labels, fontsize=7)
ax_h.set_title('Cosine Similarity Heatmap (First 20 Movies)', fontsize=12,
               color='#f0f6fc', pad=10)
plt.colorbar(im, ax=ax_h, shrink=0.8)
ax_h.set_facecolor(BG_PANEL)

# Distribution of all similarity scores
ax_d = axes[1]
upper_tri = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
ax_d.hist(upper_tri, bins=50, color=ACCENT, edgecolor='none', alpha=0.85)
ax_d.axvline(upper_tri.mean(), color=ACCENT2, linestyle='--',
             linewidth=2, label=f'Mean = {upper_tri.mean():.3f}')
ax_d.axvline(np.median(upper_tri), color=ACCENT3, linestyle=':',
             linewidth=2, label=f'Median = {np.median(upper_tri):.3f}')
ax_d.set_title('Distribution of All Pairwise Cosine Similarities', fontsize=12,
               color='#f0f6fc', pad=10)
ax_d.set_xlabel('Cosine Similarity')
ax_d.set_ylabel('Frequency')
ax_d.legend(facecolor=BG_PANEL, edgecolor='#30363d', labelcolor='#e6edf3')
ax_d.set_facecolor(BG_PANEL)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

for ax in axes:
    ax.set_facecolor(BG_PANEL)

fig2.patch.set_facecolor(BG_DARK)
plt.tight_layout()
plt.savefig('outputs/similarity_analysis.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print("  ✅ Similarity analysis saved → outputs/similarity_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating text report...")

report_lines = [
    "=" * 65,
    "  CONTENT-BASED MOVIE RECOMMENDATION SYSTEM — FULL REPORT",
    "  MovieLens-Style Dataset | TF-IDF + Cosine Similarity",
    "=" * 65,
    "",
    "── 1. DATASET OVERVIEW ─────────────────────────────────────────",
    f"  Total Movies    : {stats['total_movies']}",
    f"  Total Ratings   : {stats['total_ratings']:,}",
    f"  Total Users     : {stats['total_users']}",
    f"  Unique Genres   : {len(stats['unique_genres'])}  →  {', '.join(stats['unique_genres'][:10])}...",
    f"  Year Range      : {stats['year_range'][0]} – {stats['year_range'][1]}",
    f"  Overall Avg Rat.: {stats['avg_rating_overall']}",
    "",
    "── 2. FEATURE ENGINEERING ──────────────────────────────────────",
    "  • Genres normalized: '|' replaced with space, lowercased",
    "  • TF-IDF Vectorizer: word analyzer, unigrams + bigrams",
    f"  • TF-IDF Matrix    : {stats['tfidf_matrix_shape'][0]} movies × {stats['tfidf_matrix_shape'][1]} features",
    f"  • Cosine Sim Matrix: {stats['cosine_sim_shape'][0]} × {stats['cosine_sim_shape'][1]}",
    "  • Popularity Score : IMDB-style Bayesian weighted rating",
    "  • Year extracted   : regex from title string",
    "",
    "── 3. CONTENT-BASED RECOMMENDATIONS ────────────────────────────",
]

for title, df in rec_results.items():
    report_lines.append(f"\n  Similar to: '{title}'")
    report_lines.append(f"  {'Title':<45} {'Genres':<35} {'Sim':>6}  {'AvgRat':>6}")
    report_lines.append("  " + "─" * 96)
    for _, row in df.iterrows():
        t = row['title'][:43]
        g = row['genres'][:33]
        report_lines.append(
            f"  {t:<45} {g:<35} {row['similarity_score']:>6.3f}  {row['avg_rating']:>6.2f}"
        )

report_lines += [
    "",
    "── 4. PERSONALIZED USER RECOMMENDATIONS ────────────────────────",
]
for uid, df in user_recs.items():
    ur = ratings_df[ratings_df['userId'] == uid]
    report_lines.append(f"\n  User {uid} (rated {len(ur)} movies, avg {ur['rating'].mean():.1f}⭐):")
    for _, row in df.iterrows():
        t = row['title'][:45]
        report_lines.append(
            f"    [{row['recommendation_score']:.3f}] {t}"
        )

report_lines += [
    "",
    "── 5. GENRE-PREFERENCE RECOMMENDATIONS ─────────────────────────",
    f"  Query genres: {genre_prefs}",
    f"  {'Title':<45} {'Genres':<35} {'Score':>6}",
    "  " + "─" * 90,
]
for _, row in genre_recs.iterrows():
    t = row['title'][:43]
    g = row['genres'][:33]
    report_lines.append(f"  {t:<45} {g:<35} {row['genre_match_score']:>6.3f}")

report_lines += [
    "",
    "── 6. SIMILARITY STATISTICS ────────────────────────────────────",
    f"  Mean pairwise similarity   : {upper_tri.mean():.4f}",
    f"  Median pairwise similarity : {np.median(upper_tri):.4f}",
    f"  Max pairwise similarity    : {upper_tri.max():.4f}",
    f"  Min pairwise similarity    : {upper_tri.min():.4f}",
    f"  Std pairwise similarity    : {upper_tri.std():.4f}",
    "",
    "── 7. METHODOLOGY ──────────────────────────────────────────────",
    "  Content-Based Filtering using:",
    "  • TF-IDF on genre strings (unigrams + bigrams)",
    "  • Cosine similarity matrix between all movie pairs",
    "  • User profile: weighted avg TF-IDF vector of liked movies",
    "  • Final score: 75% content similarity + 25% popularity",
    "",
    "  Libraries: pandas, numpy, scikit-learn (TfidfVectorizer,",
    "             cosine_similarity, MinMaxScaler), matplotlib, seaborn",
    "",
    "=" * 65,
    "  END OF REPORT",
    "=" * 65,
]

report_text = "\n".join(report_lines)
with open('outputs/recommendation_report.txt', 'w') as f:
    f.write(report_text)
print(report_text)
print("\n  ✅ Report saved → outputs/recommendation_report.txt")
print("\n✅ All done! Outputs:")
print("   • outputs/eda_dashboard.png")
print("   • outputs/similarity_analysis.png")
print("   • outputs/recommendation_report.txt")
