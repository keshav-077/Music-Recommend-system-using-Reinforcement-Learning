# Music-Recommend-system-using-Reinforcement-Learning

# 🎵 Music Recommendation System using Reinforcement Learning

This project implements a music recommendation system that learns optimal song suggestions using **Reinforcement Learning (Q-Learning)**. The system is trained on a music dataset (e.g., from Spotify or Kaggle), where the agent receives rewards based on the popularity of the recommended songs. Over multiple episodes, the agent improves its recommendation strategy.

---

## 🚀 Key Features

- ✅ **Q-Learning Algorithm** to build a recommendation policy.
- 🎧 **Song Popularity-Based Reward Function**.
- 🧠 **Exploration vs. Exploitation Strategy** via Epsilon Decay.
- 📉 Tracks reward improvement across episodes.
- 🔁 Scalable to real-world user interaction and personalization.

---

## 🧠 Methodology

### 1. 📥 Data Preparation
- Load the music dataset using `pandas`.
- Normalize song features (e.g., tempo, danceability, acousticness) using `MinMaxScaler`.
- Sample 1000 songs per episode for training.

### 2. 🧪 Q-Learning Setup
- Each **state** is a discretized vector of selected song features.
- Each **action** is an index pointing to a song in the dataset.
- The **reward** is based on song popularity scaled to 0–1.
- The **Q-Table** is a nested dictionary: `Q[state][action]`.

### 3. 🔁 Training Loop
- For each episode:
  - Sample 1000 songs.
  - Choose action using epsilon-greedy policy.
  - Update Q-values based on reward and future action value.
  - Decay epsilon to balance exploration/exploitation.

```python
Q[state][action] += alpha * (
    reward + gamma * Q[next_state][best_next_action] - Q[state][action]
)
