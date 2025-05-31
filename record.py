# record_sonic_bk2.py
import retro
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    WarpFrame,
    ClipRewardEnv,
)

# Configuration
GAME_NAME = "SonicAndKnuckles-Genesis"
STATE_NAME = retro.State.DEFAULT
MODEL_DIR = "models_sonic_knuckles/"
RECORD_PATH = "recordings_sonic_knuckles/bk2_files/"
MAX_EPISODES = 1  # One Episode because sonic is deterministic
SEED = 100


def make_wrapped_env(game, state, record_path=None):
    """Create environment with training wrappers and recording"""
    env = retro.make(
        game=game,
        state=state,
        record=record_path,
        render_mode="human",  # Show gameplay
    )
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84)
    env = ClipRewardEnv(env)
    return env


def find_latest_model():
    """Find the latest trained model"""
    if not os.path.exists(MODEL_DIR):
        return None

    models = [
        f for f in os.listdir(MODEL_DIR) if f.endswith(".zip") and f.startswith("ppo_")
    ]

    if not models:
        return None

    # Sort by modification time (newest first)
    models.sort(
        key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True
    )
    return os.path.join(MODEL_DIR, models[0])


def record_agent():
    print("=== Starting BK2 Gameplay Recording ===")
    os.makedirs(RECORD_PATH, exist_ok=True)
    print(f"BK2 Output Directory: {os.path.abspath(RECORD_PATH)}")

    # Load model
    model_path = find_latest_model()
    if not model_path:
        print(f"❌ No model found in {MODEL_DIR}")
        return

    print(f"Loading model: {os.path.basename(model_path)}")
    model = PPO.load(model_path)

    # Create single environment instance with recording
    base_env = make_wrapped_env(GAME_NAME, STATE_NAME, record_path=RECORD_PATH)

    # Wrap for agent input
    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    print(f"Recording {MAX_EPISODES} episodes...")
    for episode in range(MAX_EPISODES):
        obs = vec_env.reset()
        done = [False]
        while not done[0]:
            # Get action from agent
            action, _ = model.predict(obs, deterministic=True)  # type: ignore[arg-type]

            # Step environment
            obs, _, done, _ = vec_env.step(action)

            # Render if needed
            base_env.render()

        print(f"✅ Episode {episode + 1} recorded")

    # Close environment properly
    vec_env.close()

    print("\nRecording complete. BK2 files saved to:", RECORD_PATH)
    print("Convert to video with:")
    print(
        f"python -m retro.scripts.playback_movie {os.path.join(RECORD_PATH, '*.bk2')}"
    )


if __name__ == "__main__":
    record_agent()
