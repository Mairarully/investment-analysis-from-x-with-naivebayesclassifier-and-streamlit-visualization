import tweepy
import pandas as pd
import os

from datetime import datetime, timedelta, timezone
def init_client():
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    return tweepy.Client(
        bearer_token=bearer,
        wait_on_rate_limit=True  # WAJIB TRUE
    )

def scrape_once_recent(
    client,
    keywords,
    join_year_min=2010,
    join_year_max=2022,
    days_back=5,
    max_results=100,
    lang="id",
    exclude_retweets=True
):

    # Build query
    base = " OR ".join(keywords)
    q_parts = [f"({base})"]
    if lang:
        q_parts.append(f"lang:{lang}")
    if exclude_retweets:
        q_parts.append("-is:retweet")
    query = " ".join(q_parts)
    
    # Time range (UTC) — FIX API RULE
    end_time = datetime.now(timezone.utc) - timedelta(seconds=15)
    start_time = end_time - timedelta(days=days_back)
    resp = client.search_recent_tweets(
        query=query,
        start_time=start_time,
        end_time=end_time,
        max_results=max_results,
        tweet_fields=["created_at", "author_id", "public_metrics"],
        expansions=["author_id"],
        user_fields=["created_at", "username", "name"]
    )

    if resp.data is None:
        return pd.DataFrame(), query, start_time, end_time
    users = {u.id: u for u in resp.includes.get("users", [])}
    rows = []

    for t in resp.data:
        u = users.get(t.author_id)
        if not u or not getattr(u, "created_at", None):
            continue
        join_year = pd.to_datetime(u.created_at).year
        if not (join_year_min <= join_year <= join_year_max):
            continue
        pm = t.public_metrics or {}
        rows.append({
            "tweet_id": t.id,
            "text": t.text,
            "tweet_created_at": t.created_at,
            "author_id": t.author_id,
            "username": u.username,
            "author_account_created_at": u.created_at,
            "author_join_year": join_year,
            "retweet_count": pm.get("retweet_count", 0),
            "reply_count": pm.get("reply_count", 0),
            "like_count": pm.get("like_count", 0),
            "quote_count": pm.get("quote_count", 0),
        })
    df = pd.DataFrame(rows)
    return df, query, start_time, end_time

def save_csv(df):
    if df.empty:
        print("⚠️ Data kosong, TIDAK menyimpan file.")
        return
    filename = f"tweets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} rows → {filename}")
if __name__ == "__main__":
    KEYWORDS = [
        "obligasi", "sbn", "surat berharga negara", "sukuk", "sukuk negara", "emas", "emas digital", "logam mulia"
    ]
    client = init_client()

    df, query, start_time, end_time = scrape_once_recent(
        client,
        keywords=KEYWORDS,
        join_year_min=2010,
        join_year_max=2022,
        max_results=100,
        lang="id"
    )

    print("Query digunakan:")
    print(query)
    print("Rentang waktu:", start_time, "->", end_time)
    print("Jumlah data lolos filter:", len(df))

    save_csv(df)