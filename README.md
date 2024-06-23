# Deep-Trade-AI---Q-Learning-Trading-Bot


# Overview

This project implements a Stock Trading Bot, trained using Deep Reinforcement Learning, specifically Deep Q-learning. Implementation is kept simple and as close as possible to the algorithm discussed in the paper, for learning purposes.

## Introduction

Generally, Reinforcement Learning is a family of machine learning techniques that allow us to create intelligent agents that learn from the environment by interacting with it, as they learn an optimal policy by trial and error. This is especially useful in many real world tasks where supervised learning might not be the best approach due to various reasons like nature of task itself, lack of appropriate labelled data, etc.

The important idea here is that this technique can be applied to any real world task that can be described loosely as a Markovian process.

## Approach

This work uses a Model-free Reinforcement Learning technique called Deep Q-Learning (neural variant of Q-Learning).
At any given time (episode), an agent abserves it's current state (n-day window stock price representation), selects and performs an action (buy/sell/hold), observes a subsequent state, receives some reward signal (difference in portfolio position) and lastly adjusts it's parameters based on the gradient of the loss computed.



## Results

Trained on `GOOG` 2010-17 stock data, tested on 2019 with a profit of $1141.45 with a model accuracy of 89.98% :

![Google Stock Trading episode](./extra/visualization.png)

You can obtain similar visualizations of your model evaluations using the [notebook](./visualize.ipynb) provided.



## Data

You can download Historical Financial data from [Yahoo! Finance](https://ca.finance.yahoo.com/) for training, or even use some sample datasets already present under `data/`.

## Getting Started

In order to use this project, you'll need to install the required python packages:

```bash
pip3 install -r requirements.txt
```

Now you can open up a terminal and start training the agent:

```bash
python3 train.py data/GOOG.csv data/GOOG_2018.csv --strategy t-dqn
```

Once you're done training, run the evaluation script and let the agent make trading decisions:

```bash
python3 eval.py data/GOOG_2019.csv --model-name model_GOOG_50 --debug
```

Now you are all set up!

## Acknowledgements

- [@keon](https://github.com/keon) for [deep-q-learning](https://github.com/keon/deep-q-learning)
- [@edwardhdlu](https://github.com/edwardhdlu) for [q-trader](https://github.com/edwardhdlu/q-trader)

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
