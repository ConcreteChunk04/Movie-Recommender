"""
Movie Recommendation System — Collaborative Filtering on REAL MovieLens data
============================================================================

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_FACTORS = 20


def load_movielens():
    """
    Loads real ratings.csv (userId, movieId, rating, timestamp) and
    movies.csv (movieId, title, genres).

    Real MovieLens quirks we must handle:
      - movieId values are NOT contiguous (e.g. 1, 5, 87, 9012) → remap to 0..N
      - ratings are in 0.5 steps from 0.5 to 5.0
    """
    ratings = pd.read_csv("dataset/ratings.csv")
    movies  = pd.read_csv("dataset/movies.csv")

    # Remap user & movie IDs to contiguous integer indices for the matrix
    user_ids  = {uid: i for i, uid in enumerate(ratings.userId.unique())}
    movie_ids = {mid: i for i, mid in enumerate(ratings.movieId.unique())}

    ratings["u_idx"] = ratings.userId.map(user_ids)
    ratings["m_idx"] = ratings.movieId.map(movie_ids)

    # Lookup table: matrix index → movie title/genre
    idx_to_movie = (
        movies.set_index("movieId")
        .reindex(movie_ids.keys())
        .assign(m_idx=list(movie_ids.values()))
        .set_index("m_idx")
    )

    n_users  = len(user_ids)
    n_movies = len(movie_ids)
    return ratings, idx_to_movie, n_users, n_movies


def build_and_predict(train_df, n_users, n_movies):
    mat = train_df.pivot_table(
        index="u_idx", columns="m_idx", values="rating"
    ).reindex(index=range(n_users), columns=range(n_movies))

    user_means = mat.mean(axis=1).fillna(mat.stack().mean())
    mat_filled = mat.apply(lambda col: col.fillna(user_means), axis=0)

    R      = mat_filled.values
    R_norm = R - user_means.values[:, np.newaxis]

    U, sigma, Vt = svds(R_norm, k=N_FACTORS)
    R_pred = np.dot(U, np.dot(np.diag(sigma), Vt)) + user_means.values[:, np.newaxis]
    R_pred = np.clip(R_pred, 0.5, 5.0)

    return pd.DataFrame(R_pred, index=mat.index, columns=mat.columns), mat, user_means


def evaluate(pred_df, test_df, user_means):
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        u, m, r = int(row.u_idx), int(row.m_idx), row.rating
        if u in pred_df.index and m in pred_df.columns:
            preds.append(float(pred_df.loc[u, m]))
            actuals.append(r)

    preds, actuals = np.array(preds), np.array(actuals)
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    mae  = np.mean(np.abs(preds - actuals))

    base = [float(user_means.loc[int(r.u_idx)]) for _, r in test_df.iterrows()
            if int(r.u_idx) in user_means.index]
    base_rmse = np.sqrt(np.mean((np.array(base) - actuals) ** 2))

    print(f"\nSVD Model   — RMSE: {rmse:.3f}  |  MAE: {mae:.3f}")
    print(f"User-Mean   — RMSE: {base_rmse:.3f}")
    print(f"Improvement:  {(base_rmse-rmse)/base_rmse*100:.1f}% lower RMSE vs baseline")
    return rmse, mae, base_rmse


def top_n_recs(u_idx, pred_df, actual_matrix, idx_to_movie, n=5):
    rated  = set(actual_matrix.loc[u_idx].dropna().index)
    unseen = [m for m in pred_df.columns if m not in rated]
    scores = pred_df.loc[u_idx, unseen].sort_values(ascending=False).head(n)
    print(f"\nTop {n} recommendations for User index {u_idx}:")
    for m_idx, score in scores.items():
        title = idx_to_movie.loc[m_idx, "title"]
        print(f"  {title}  →  predicted: {score:.2f} ⭐")



def make_charts(ratings, rmse, base_rmse):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#1F4E79", edgecolor="white", width=0.35)
    ax.set_xlabel("Star Rating", fontsize=12)
    ax.set_ylabel("Number of Ratings", fontsize=12)
    ax.set_title("Distribution of MovieLens Ratings", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    labs = ["User-Mean\nBaseline", f"SVD (k={N_FACTORS})"]
    vals = [base_rmse, rmse]
    b2 = ax.bar(labs, vals, color=["#BBBBBB","#1F4E79"], width=0.4, edgecolor="white")
    for bar, v in zip(b2, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    pct = (base_rmse - rmse) / base_rmse * 100
    ax.text(0.5, 0.93, f"↓ {pct:.1f}% lower RMSE than baseline", transform=ax.transAxes,
            ha="center", fontsize=10, color="#1F4E79", style="italic")
    ax.set_ylabel("RMSE (lower is better)", fontsize=12)
    ax.set_title("SVD vs Baseline RMSE", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.3)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2.0)
    fig.savefig("recommendation_charts.png", dpi=140)
    plt.close()
    print("\nSaved: recommendation_charts.png")


def main():
    print("="*60)
    print("Movie Recommendation System — REAL MovieLens Data (SVD)")
    print("="*60)

    ratings, idx_to_movie, n_users, n_movies = load_movielens()
    sparsity = 1 - len(ratings) / (n_users * n_movies)
    print(f"\nDataset: {len(ratings):,} ratings | {n_users} users | "
          f"{n_movies} movies | Sparsity: {sparsity:.1%}")

    test_df  = ratings.sample(frac=0.2, random_state=RANDOM_SEED)
    train_df = ratings.drop(test_df.index)

    pred_df, actual_matrix, user_means = build_and_predict(train_df, n_users, n_movies)
    rmse, mae, base_rmse = evaluate(pred_df, test_df, user_means)
    top_n_recs(0, pred_df, actual_matrix, idx_to_movie)
    make_charts(ratings, rmse, base_rmse)
    print("\nDone.")


if __name__ == "__main__":
    main()
