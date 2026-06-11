# Fix for Issue #1: Implement logging for training metrics

**Fix for Issue #1: Implement logging for training metrics**
===========================================================

### Overview

To implement logging for training metrics, we will add a simple logger to track episodic rewards and optionally save logs to a file or integrate with a tool like TensorBoard or WandB. Since the relevant code files (`src/main.py`, `src/agent.py`, `src/trainer.py`) are not available, we will provide a general solution that can be adapted to the existing codebase.

### Modified Files

We will create a new file `src/logger.py` to handle logging and modify `src/trainer.py` to use the logger.

### Step-by-Step Implementation

#### Step 1: Create a new file `src/logger.py`

```python
# src/logger.py
import logging
import os

class Logger:
    def __init__(self, log_file=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_file:
            self.file_handler = logging.FileHandler(log_file)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def log(self, message):
        self.logger.info(message)
```

#### Step 2: Modify `src/trainer.py` to use the logger

```python
# src/trainer.py
from src.logger import Logger

class Trainer:
    def __init__(self, log_file=None):
        self.logger = Logger(log_file)

    def train(self, agent, episodes):
        for episode in range(episodes):
            # Train the agent for one episode
            reward = agent.train_episode()

            # Log the episodic reward
            self.logger.log(f'Episode {episode+1}, Reward: {reward}')
```

#### Step 3: Optionally integrate with TensorBoard or WandB

To integrate with TensorBoard or WandB, you can use their respective APIs to log metrics. For example, with TensorBoard:

```python
# src/trainer.py
import tensorflow as tf

class Trainer:
    def __init__(self, log_file=None):
        self.logger = Logger(log_file)
        self.writer = tf.summary.create_file_writer('logs')

    def train(self, agent, episodes):
        for episode in range(episodes):
            # Train the agent for one episode
            reward = agent.train_episode()

            # Log the episodic reward to TensorBoard
            with self.writer.as_default():
                tf.summary.scalar('reward', reward, step=episode)

            # Log the episodic reward to the console/file
            self.logger.log(f'Episode {episode+1}, Reward: {reward}')
```

### Commit Message

`feat: Implement logging for training metrics`

### Example Use Case

To use the logger, create a `Trainer` instance and pass the `log_file` argument to save logs to a file:
```python
trainer = Trainer(log_file='logs/training.log')
trainer.train(agent, episodes=100)
```
This will log the episodic rewards to the console and save them to the `logs/training.log` file.