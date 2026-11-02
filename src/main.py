"""
Command line runner for the original VibeMatcher weighted recommender.

Usage:
    python -m src.main
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Example user profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for song, score, explanation in recommendations:
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


def test_adversarial_profiles() -> None:
    """Print recommendations for adversarial user profiles (manual check)."""
    songs = load_songs("data/songs.csv")

    # Adversarial profiles: conflicting preferences or unrealistic combinations
    profiles = [
        ("High energy pop with sad mood (conflicting)", {"genre": "pop", "mood": "sad", "energy": 0.9, "likes_acoustic": False}),
        ("Low energy rock with chill mood (conflicting)", {"genre": "rock", "mood": "chill", "energy": 0.1, "likes_acoustic": False}),
        ("Non-existent genre/mood with acoustic preference", {"genre": "nonexistent", "mood": "nonexistent", "energy": 0.5, "likes_acoustic": True}),
    ]

    for name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print(f"\n{name}:")
        for song, score, explanation in recommendations:
            print(f"{song['title']} - Score: {score:.2f} - {explanation}")
        print()


if __name__ == "__main__":
    main()
    # test_adversarial_profiles()
