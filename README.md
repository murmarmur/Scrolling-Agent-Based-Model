Agent-Based Scrolling Behaviour Simulation
===========================================

Overview
--------
This project implements an agent-based computational model designed to simulate user scrolling behaviour on digital platforms. The model abstracts core psychological mechanisms - boredom, novelty-seeking, and reward processing - to isolate the factors that contribute to excessive scrolling behaviour. It provides an exploratory tool for studying user engagement and can serve as a scalable foundation for future research.

Model Description
-----------------
The simulation consists of two main agent types:

- **Consumer Agents:**  
  Represent individual users interacting with digital content. Each consumer has a unique interest profile, a preferred lag (which reflects the sequence of content they engage with), and a boredom threshold. Consumers may be static (fixed interests) or dynamic (interests that change over time). Each consumer is also assigned a novelty sensitivity parameter, which determines how strongly they react to novel content.

- **Producer Agents:**  
  Represent content providers (e.g., digital platforms) competing to capture consumer attention. Producers deliver content using a strategy that adapts to consumer feedback. Their performance is measured by a rating that is adjusted based on consumer actions such as clicks, scrolls, or leaving due to boredom.

Key Mechanisms
--------------
1. **Novelty Utility:**  
   - Computes a score reflecting the novelty of content and the consumer’s tendency to seek fresh material.  
   - Based on the proportion of content items that are active and unknown to the consumer, scaled by their individual novelty sensitivity.

2. **Reward-Based Utility:**  
   - Models the effect of reward timing on engagement using a Gaussian function.  
   - Each consumer has an ideal time interval between rewards (mean, μ) and a tolerance (standard deviation, σ).

3. **Content Delivery Strategy:**  
   - Producers use a moving window strategy to display content. The window, controlled by a simulation step parameter, shifts over the available topics.  
   - Consumer feedback allows the producer’s knowledge to be updated, which reduces the number of "unknown" topics available in future cycles.

4. **Feedback and Boredom:**  
   - Consumer feedback (e.g., "click," "scroll," "leave," "hooked") alters their state.  
   - Boredom decreases as a function of reward and novelty utilities, but if boredom exceeds a threshold, the consumer disengages and may switch to another producer.

Software Architecture
---------------------
The model is implemented in Python using the following libraries:
- **NumPy:** For numerical computations.
- **Tkinter:** For the graphical user interface (GUI), which displays real-time simulation data.
- **Standard Libraries:** (csv, threading, datetime, etc.) for data handling, logging, and simulation control.

The simulation utilizes a CSV file ('consumers.csv') to load or generate consumer profiles and logs simulation events to a CSV log file ('simulation_log.csv') for later analysis.

Installation and Usage
----------------------
**Prerequisites:**  
- Python 3.10.7 
- Required Libraries: NumPy, Tkinter, Threading, Time, Random, Csv, Os, Ast, Datetime (typically bundled with Python)

**Installation:**  
If needed, install NumPy and Tkinter via pip:  
```bash
pip install numpy

pip install tkinter
