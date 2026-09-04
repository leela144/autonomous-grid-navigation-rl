# 🚕 Autonomous Grid Navigation using Reinforcement Learning

A reinforcement-learning based autonomous navigation prototype that learns efficient movement and task-completion decisions in a simulated grid environment.

The system uses Q-Learning with the Gymnasium Taxi-v4 environment, where an agent learns to navigate through a constrained 5×5 grid, pick up a passenger, and deliver the passenger to a destination.

The project also evaluates the learned policy against a random-action baseline and a BFS-based shortest-path benchmark, and provides an interactive Streamlit dashboard for route visualization and performance analysis.

---

## 🚀 Features

- Reinforcement Learning based navigation
- Q-Learning agent
- Gymnasium Taxi-v4 environment
- Discrete 5×5 grid navigation
- State and action space modeling
- Epsilon-greedy exploration
- Q-table based policy learning
- Reward-driven learning
- Custom taxi task configuration
- Passenger pickup and destination drop-off
- Learned route execution
- BFS shortest-path benchmark
- Random-agent baseline
- Route efficiency calculation
- Success-rate evaluation
- Training reward analysis
- Steps-per-episode analysis
- Epsilon decay tracking
- Hyperparameter experiments
- Interactive Streamlit dashboard
- Taxi route visualization
- Actual Taxi-v4 movement barriers visualization

---

## 🎯 Project Objective

The goal of this project is to demonstrate how Reinforcement Learning can be used for sequential decision making in an autonomous navigation scenario.

Instead of directly using real-world GPS or road-network data, the project uses the Taxi-v4 environment as a controlled simulation.

The agent learns which action to take based on its current state, receives rewards or penalties from the environment, and gradually learns an efficient navigation policy through repeated interaction.

The project focuses specifically on the reinforcement-learning decision-making component of autonomous navigation.

---

## 🏗️ Architecture

```text
Taxi-v4 Environment
        │
        ▼
State Representation
        │
        ▼
Q-Learning Agent
        │
        ▼
Training Engine
        │
        ▼
Trained Q-Table
        │
        ├───────────────┐
        ▼               ▼
Evaluation        Route Execution
        │               │
        ▼               ▼
Baselines        Route Visualization
        │               │
        └───────┬───────┘
                ▼
        Performance Analysis
                │
                ▼
        Streamlit Dashboard
```
---
## 🎮 Environment

The project uses the Gymnasium Taxi-v4 environment.

Taxi-v4 represents a simplified grid-based navigation problem.

The environment contains:

-A 5×5 grid
-Four landmark locations
-A taxi agent
-A passenger
-A destination
-Movement barriers
-Pickup and drop-off actions
-Rewards and penalties

The taxi must:

-Navigate through the environment
-Reach the passenger
-Pick up the passenger
-Navigate to the destination
-Drop off the passenger

---

## 📊 State Space
 
Taxi-v4 contains: 500 possible states

Each state represents:

-Taxi row
-Taxi column
-Passenger location
-Destination location

---
## 🎯 Action Space

The environment provides six possible actions:

Action	Meaning
0	South
1	North
2	East
3	West
4	Pickup
5	Dropoff

---
## 🏆 Reward System

The environment provides rewards based on the agent's actions.

Typical rewards include:

Valid movement        → -1
Blocked movement      → -1
Incorrect pickup      → -10
Successful drop-off   → +20

---
## 📈 Exploration and Exploitation

```text
The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

Exploration
     ↓
Exploration + Exploitation
     ↓
Mostly Exploitation
     ↓
Learned Policy

```
---

## 📂 Project Structure
```text
autonomous-grid-navigation-rl/
│
├── agent/
│   ├── q_learning.py
│   └── route_executor.py
│
├── environment/
│   ├── taxi_environment.py
│   └── route_planner.py
│
├── training/
│   └── trainer.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── random_baseline.py
│   ├── metrics.py
│   └── route_comparison.py
│
├── experiments/
│   └── hyperparameter_search.py
│
├── visualization/
│   └── plots.py
│
├── models/
│   └── q_table.npy
│
├── results/
│   ├── rewards.npy
│   ├── steps.npy
│   ├── epsilon.npy
│   └── hyperparameter_results.csv
│
├── tests/
│   ├── test_taxi.py
│   ├── test_agent.py
│   ├── test_custom_environment.py
│   ├── test_route_planner.py
│   ├── test_visualization.py
│   ├── test_route_executor.py
│   └── test_route_comparison.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_model.py
│   ├── evaluate_random.py
│   ├── compare_agents.py
│   └── analyze_training.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/leela144/autonomous-grid-navigation-rl.git
cd autonomous-grid-navigation-rl
```
### 2. Create a virtual environment
```bash
python -m venv venv
```
### 3. Activate the environment
```bash
venv\Scripts\activate
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
---
## 🧭 Route Planning

The project includes a BFS-based route planner that is used as a shortest-path benchmark.

Breadth-First Search explores valid environment transitions and determines a shortest valid action sequence for a given taxi task.

The learned Q-Learning route can then be compared against this benchmark.
 ```text
Custom Taxi Task
       ↓
Q-Learning Route
       ↓
BFS Shortest Route
       ↓
Performance Comparison
```
---
## 🧠 Q-Learning

The project uses Q-Learning, a model-free Reinforcement Learning algorithm.

The agent maintains a Q-table that stores the expected value of taking each action from every state.

The Q-table has:

```text
500 states × 6 actions

The Q-Learning update rule is:
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```
---
## 📐 Route Efficiency

Route efficiency is calculated by comparing the learned route length with the shortest valid route.

Route Efficiency =
Optimal Steps / RL Steps × 100
---
## 🖥️ Streamlit Dashboard
```bash
streamlit run app.py
```
## 📊 Dashboard Performance Metrics

The Streamlit dashboard displays:

-RL Agent Steps
-Optimal Steps
-Extra Steps
-Route Efficiency
-Task Success
-Route Optimality
-Learned Action Sequence
-Route Coordinates
---
## 🛠️ Technologies Used
 -Python
-Gymnasium
-NumPy
-Pandas
-Matplotlib
-Streamlit
-Q-Learning
-Breadth-First Search (BFS)
-Reinforcement Learning
---
## 📚 Learning Outcomes

This project provided practical experience with:

-Reinforcement Learning
-Q-Learning
-Markov Decision Processes
-Exploration vs exploitation
-Reward engineering
-State-action modeling
-Policy learning
-Training analysis
-Baseline comparison
-Shortest-path algorithms
-Hyperparameter experimentation
-Streamlit application development
-Modular Python project design
---