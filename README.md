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
**Graphical User Interface (GUI):**
- In addition to the core simulation mechanics, the project features a real-time GUI built with Tkinter. The GUI not only displays dynamic producer-consumer interactions and simulation states but also provides interactive controls, such as start and stop buttons, to facilitate user engagement with the simulation. The interface refreshes every 100 milliseconds, ensuring that the visualised data accurately reflects the current state of the simulation. The simulation automatically stops when all consumers achieve the "hooked" state.

The model is implemented in Python using the following libraries:
- **NumPy:** For numerical computations.
- **Tkinter:** For the graphical user interface (GUI), which displays real-time simulation data.
- **Standard Libraries:** (csv, threading, datetime, etc.) for data handling, logging, and simulation control.

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
```
Data Files and Dependencies
----------------------
- **names-list.txt**  
  This file was sourced from [Repository "random-name"](https://github.com/dominictarr/random-name/blob/master/first-names.txt) for generating consumer names.

- **consumers.csv** and **simulation_log.csv**  
  These files are used by the model to store consumer data and log simulation events. If they do not exist, they will be created automatically when the simulation runs.  
  The number of consumers can be adjusted in the `load_consumers_from_csv()` method (specifically in the `generate_random_consumers()` function).

**Example File Contents:**

- **names-list.txt:**
```bash
Name,Interests,Lag,BoredomThreshold,Producer,Type,Novelty,RewardType
Nesta,"[0, 0, 0, 1, 0, 0, 1, 1, 0, 1]",2,19,Producer 1,static,0.53,Variable SD
Tani,"[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]",2,23,Producer 2,dynamic,0.98,Variable SD
Morna,"[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]",4,23,Producer 1,dynamic,0.64,Variable Both
Gwenora,"[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]",5,13,Producer 2,static,1.08,Variable Mean
Zandra,"[0, 0, 1, 0, 0, 0, 1, 0, 0, 0]",2,20,Producer 1,static,0.95,Variable Mean
Miguela,"[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]",5,22,Producer 2,static,1.11,Fixed
```

- **consumers.csv:**
```bash
timestamp,producer,consumer,cycle,phase,current_lag,knowledge,feedback,boredom_threshold,current_boredom,interests,consumer_type,reward_function_type,interest changed,hooked_producer
2025-04-16 14:13:14,Producer 1,Nesta,1,discovery,0,"U,U,U,U,U,U,U,U,U,U",Starting cycle,19,0,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:14,Producer 1,Nesta,1,discovery,0,"N,N,N,U,U,U,U,U,U,U",scroll,19,3,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:14,Producer 1,Nesta,1,discovery,0,"N,N,N,M,M,M,U,U,U,U",click,19,4.840418461763329,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:15,Producer 1,Nesta,1,discovery,0,"N,N,N,M,M,M,M,M,M,U",click,19,5.681410787637234,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:15,Producer 1,Nesta,1,discovery,0,"N,N,N,M,M,M,M,M,M,I",click,19,7.522407092711602,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:16,Producer 1,Nesta,1,discovery,0,"N,N,N,I,M,M,M,M,M,I",click,19,9.522403281492824,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:16,Producer 1,Nesta,1,discovery,0,"N,N,N,I,N,M,M,M,M,I",scroll,19,12.522403281492824,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:17,Producer 1,Nesta,1,discovery,0,"N,N,N,I,N,N,M,M,M,I",scroll,19,15.522403281492824,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:17,Producer 1,Nesta,1,discovery,0,"N,N,N,I,N,N,I,M,M,I",click,19,17.47789712597146,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:18,Producer 1,Nesta,1,discovery,0,"N,N,N,I,N,N,I,M,M,I",leave,19,19.477893477272335,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:18,Producer 1,Nesta,1,discovery,0,"N,N,N,I,N,N,I,M,M,I",left due to boredom,19,19.477893477272335,"0,0,0,1,0,0,1,1,0,1",static,Variable SD,No,
2025-04-16 14:13:18,Producer 1,Morna,1,discovery,0,"U,U,U,U,U,U,U,U,U,U",Starting cycle,23,0,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:18,Producer 1,Morna,1,discovery,0,"N,N,N,U,U,U,U,U,U,U",scroll,23,3,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:18,Producer 1,Morna,1,discovery,0,"N,N,N,M,M,M,U,U,U,U",click,23,0,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:19,Producer 1,Morna,1,discovery,0,"N,N,N,M,M,M,N,N,N,U",scroll,23,3,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:19,Producer 1,Morna,1,discovery,0,"N,N,N,M,M,M,N,N,N,N",scroll,23,6,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:20,Producer 1,Morna,1,discovery,0,"N,N,N,N,M,M,N,N,N,N",scroll,23,9,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:20,Producer 1,Morna,1,discovery,0,"N,N,N,N,I,M,N,N,N,N",click,23,0,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:21,Producer 1,Morna,1,discovery,0,"N,N,N,N,I,N,N,N,N,N",scroll,23,3,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:21,Producer 1,Morna,1,discovery,0,"N,N,N,N,I,N,N,N,N,N",click,23,0,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:22,Producer 1,Morna,1,discovery,0,"N,N,N,N,I,N,N,N,N,N",click,23,0,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:22,Producer 1,Morna,1,discovery,0,"N,N,N,N,I,N,N,N,N,N",click,23,0,"0,0,0,0,1,0,0,0,0,0",dynamic,Variable Both,No,
2025-04-16 14:13:23,Producer 1,Morna,1,discovery,0,"N,N,I,N,N,N,N,N,N,N",click,23,0,"0,0,1,0,0,0,0,0,0,0",dynamic,Variable Both,Yes,
```
