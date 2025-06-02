import retro
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    WarpFrame,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration
GAME_NAME = "SonicAndKnuckles-Genesis"
STATE_NAME = retro.State.DEFAULT

# Directories
LOG_DIR = "logs_sonic_knuckles/"
MODEL_DIR = "models_sonic_knuckles/"
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "ppo_tensorboard/")

# Training Hyperparameters
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 2.5e-4
N_STEPS = 128
BATCH_SIZE = 32
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.1


def make_env(game, state, seed=0, monitor_log_dir=None, render_mode=None):
    """Create a wrapped environment with simplified recording logic"""
    # Create environment
    # Reward is defined in ./scenario.json
    env = retro.make(
        game=game, state=state, render_mode=render_mode, scenario="./scenario.json"
    )

    # Apply standard wrappers
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84)

    # Add monitoring
    if monitor_log_dir:
        os.makedirs(monitor_log_dir, exist_ok=True)
        env = Monitor(env, monitor_log_dir, info_keywords=("score", "lives", "rings"))

    return env


def main():
    print(f"Starting training for {GAME_NAME}")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    print("Setting up training environment...")
    env = DummyVecEnv(
        [lambda: make_env(GAME_NAME, STATE_NAME, seed=0, monitor_log_dir=LOG_DIR)]
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    save_freq = max(25000 // env.num_envs, 1)
    print(f"Checkpoint save frequency: {save_freq} steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=MODEL_DIR,
        name_prefix=f"ppo_{GAME_NAME.replace('-', '_').lower()}",
    )

    print("Initializing PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        policy_kwargs=dict(normalize_images=True),
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    print(f"TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    print(f"Run: tensorboard --logdir {TENSORBOARD_LOG_DIR}")

    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    end_time = time.time()

    print(
        f"Training completed in {end_time - start_time:.2f} seconds ({(end_time - start_time) / 60:.2f} minutes)"
    )

    # Save final model with timestep information
    final_model_path = os.path.join(
        MODEL_DIR,
        f"ppo_{GAME_NAME.replace('-', '_').lower()}_final_{TOTAL_TIMESTEPS}.zip",
    )
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    print("Training environment closed.")


if __name__ == "__main__":
    main()
