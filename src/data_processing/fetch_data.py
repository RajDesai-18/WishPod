import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def get_access_token():
    """Authenticate with Spotify API and return access token."""
    auth_url = "https://accounts.spotify.com/api/token"
    response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    }, timeout=10)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Failed to authenticate: {response.json()}")


def fetch_podcasts(query, limit=50):
    """Fetch podcasts based on a query using Spotify's Search API."""
    token = get_access_token()
    url = f"https://api.spotify.com/v1/search?q={query}&type=show&limit={limit}"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 1))
        print(f"Rate limit hit. Retrying after {retry_after} seconds...")
        time.sleep(retry_after)
        return fetch_podcasts(query, limit)
    else:
        raise Exception(f"Failed to fetch podcasts: {response.json()}")


def extract_podcast_data(data):
    """Extract podcast data from the API response."""
    shows = data.get('shows', {}).get('items', [])
    podcasts = []
    for show in shows:
        podcasts.append({
            'id': show['id'],
            'name': show['name'],
            'description': show['description'],
            'publisher': show['publisher'],
            'total_episodes': show['total_episodes'],
            'explicit': show['explicit'],
            'popularity': show.get('popularity', 0),
            'external_url': show['external_urls'].get('spotify', ''),
            'language': show.get('languages', [None])[0],
            'image_url': show['images'][0]['url'] if show.get('images') else None
        })
    return podcasts


if __name__ == "__main__":
    # Define genres and their respective limits
    genres_with_limits = {
        "technology": 50,
        "comedy": 45,
        "news": 40,
        "education": 50,
        "health": 35,
        "sports": 40,
        "horror": 25,
        "crime": 30,
        "lifestyle": 25,
        "food": 35,
        "Spiritual and devotional": 15,
        "stories": 30
    }

    all_podcasts = []

    for genre, limit in genres_with_limits.items():
        print(f"Fetching {limit} podcasts for genre: {genre}")
        try:
            response = fetch_podcasts(query=genre, limit=limit)
            podcasts = extract_podcast_data(response)
            for podcast in podcasts:
                podcast['genre'] = genre  # Add genre to each podcast
            all_podcasts.extend(podcasts)
        except Exception as e:
            print(f"Error fetching podcasts for genre {genre}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_podcasts)

    # Save to CSV
    output_path = "D:/Classes/Machine Learning/Project/Wishpod/data/raw/podcasts.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Podcasts data saved to {output_path}")
