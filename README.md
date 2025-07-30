This project uses stable-retro, a modern fork of openai gym-retro

# 🦔 Sonic the Hedgehog AI Agent with Stable Baselines3

![Sonic Gameplay Demo](gameplay.gif)  
_Agent playing Sonic after 1 million training steps_

This project trains an AI agent to play Sonic the Hedgehog using Proximal Policy Optimization (PPO) from Stable Baselines3. The agent was trained for 1 million timesteps on the Sonic and Knuckles Genesis game.

A video of an unedited run with three lifes can be found [here](recordings_sonic_knuckles/output.mp4)

## 📦 Installation

This project uses python 3.10

### Using uv

```bash
uv venv -p 3.10 # Creates virtual environment (Python 3.10)
source .venv/bin/activate  # Linux/Mac
uv sync
```

### Using requirements.txt

```bash
pip install -r requirements.txt
```

After installing stable-retro, import the rom:
```bash
python -m retro.import ./rom
```

## 🚀 Usage

### Training

```bash
python training.py
```

### Recording / Viewing Gameplay

```bash
python record.py
```
