import tkinter as tk
import threading
import numpy as np
import time
import random
import csv
import os
import ast
import datetime

#############################################
# Global Simulation Control & Shared State
#############################################

simulation_running = False

# A global dictionary to hold the latest visual state for each session.
# Keys: (producer_name, consumer_name); Values: dict with keys:
# cycle, phase, current_lag, knowledge, latest_feedback, boredom_threshold, current_boredom,
# interests, consumer_type, reward_function_type, interest changed, hooked_producer.
visual_state = {}
state_lock = threading.Lock()

# Global dictionary to map consumer names to consumer objects (for swapping purposes)
global_consumers = {}

# Initialise log file with header if it doesn't exist.
LOG_FILE = "simulation_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow([
            "timestamp", "producer", "consumer", "cycle", "phase", "current_lag",
            "knowledge", "feedback", "boredom_threshold", "current_boredom",
            "interests", "consumer_type", "reward_function_type",
            "interest changed", "hooked_producer"
        ])


def log_simulation_event(producer, consumer, cycle, phase, current_lag, knowledge, feedback,
                         boredom_threshold, current_boredom, interests, consumer_type,
                         reward_function_type, interest_changed, hooked_producer):
    """
        Logs a simulation event by appending a record to the CSV log file, allows a post simulation data analysis.

        Parameters:
            producer (str): The name of the default producer (assigned to a consumer at the start of the simulation).
            consumer (str): The name of the consumer.
            cycle (int): The current simulation cycle.
            phase (str): The current simulation phase ("discovery" or "lag").
            current_lag (int): The current lag value implemented by a producer.
            knowledge (list of str): The producer's knowledge state of consumer's interests.
            feedback (str): Consumer's feedback at this timestep: "click", "scroll", "leave", or "hooked".
            boredom_threshold (int): The threshold value at which a consumer leaves.
            current_boredom (int): The consumer's current boredom level.
            interests (list of int): A binary list representing consumer's interests.
            consumer_type (str): The type of consumer ("dynamic" or "static").
            reward_function_type (str): The preferred type of reward function assigned to consumer.
            interest_changed (str): Indicates whether an interest change occurred ("Yes" or "No") - applies only to dynamic consumers.
            hooked_producer (str): The name of the producer with whom the consumer reached "hooked" state; otherwise empty.
        """
    with open(LOG_FILE, "a", newline="") as logfile:
        writer = csv.writer(logfile)
        knowledge_str = ",".join(knowledge)
        interests_str = ",".join(map(str, interests))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, producer, consumer, cycle, phase, current_lag,
                         knowledge_str, feedback, boredom_threshold, current_boredom, interests_str,
                         consumer_type, reward_function_type, interest_changed, hooked_producer])


def update_visual_state(prod_name, cons_name, cycle, phase, current_lag, knowledge, latest_feedback, consumer):
    """
        Updates the shared visual state dictionary with the current simulation parameters and logs the event.

        Parameters:
            prod_name (str): The name of the producer.
            cons_name (str): The name of the consumer.
            cycle (int): The current simulation cycle.
            phase (str): The current simulation phase.
            current_lag (int): The lag value for the current cycle.
            knowledge (list of str): The updated knowledge state for each content item.
            latest_feedback (str): The latest feedback signal (e.g., "click", "scroll", "leave", "hooked").
            consumer (Consumer): The consumer instance (used to extract additional attributes such as boredom, interests, etc.).

        """
    consumer_type = "dynamic" if consumer.is_dynamic else "static"
    reward_function_type = consumer.reward_function_type
    interest_changed = "Yes" if consumer.interest_changed else "No"
    consumer.interest_changed = False
    hooked_producer = prod_name if latest_feedback.lower() == "hooked" else ""
    with state_lock:
        visual_state[(prod_name, cons_name)] = {
            "cycle": cycle,
            "phase": phase,
            "current_lag": current_lag,
            "knowledge": knowledge.copy(),
            "latest_feedback": latest_feedback,
            "boredom_threshold": consumer.boredom_threshold,
            "current_boredom": consumer.boredom,
            "interests": consumer.true_interests.copy(),
            "consumer_type": consumer_type,
            "reward_function_type": reward_function_type,
            "interest changed": interest_changed,
            "hooked_producer": hooked_producer
        }
    log_simulation_event(prod_name, cons_name, cycle, phase, current_lag, knowledge,
                         latest_feedback, consumer.boredom_threshold, consumer.boredom,
                         consumer.true_interests, consumer_type, reward_function_type,
                         interest_changed, hooked_producer)


#############################################
# Producer Competition Helper Function
#############################################

def choose_producer_weighted(producers):
    """
    Selects a producer from a list using weighted probability based on each producer's rating.

    Parameters:
        producers (list of Producer): A list of producer instances.

    Returns:
        Producer: The chosen producer based on the weighted random selection.
    """
    total = sum(p.rating for p in producers)
    rnd = random.uniform(0, total)
    upto = 0
    for p in producers:
        if upto + p.rating >= rnd:
            return p
        upto += p.rating
    return producers[-1]


#############################################
# CSV Consumer Generation & Loading Functions
#############################################

def generate_random_consumers(num_consumers):
    """
        Generates a CSV file containing random consumer data.

        Parameters:
            num_consumers (int): The number of consumer records to generate.

        """
    with open('first-names.txt', 'r') as f:
        names = f.read().splitlines()
    consumers_data = []
    reward_types = ["Fixed", "Variable Mean", "Variable SD", "Variable Both"]
    for i in range(num_consumers):
        name = random.choice(names)
        interests = [0] * 10
        num_interests = random.randint(1, 5)
        chosen_indices = random.sample(range(10), num_interests)
        for idx in chosen_indices:
            interests[idx] = 1
        lag = random.randint(0, 5)
        boredom_threshold = random.randint(7, 25)
        producer = "Producer 1" if i % 2 == 0 else "Producer 2"
        consumer_type = random.choice(["dynamic", "static"])
        novelty_sensitivity = round(random.uniform(0.5, 1.5), 2)
        reward_function_type = random.choice(reward_types)
        consumers_data.append([name, interests, lag, boredom_threshold, producer, consumer_type,
                               novelty_sensitivity, reward_function_type])
    with open('consumers.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Interests', 'Lag', 'BoredomThreshold', 'Producer', 'Type', 'Novelty', 'RewardType'])
        for data in consumers_data:
            writer.writerow([data[0], str(data[1]), data[2], data[3], data[4], data[5], data[6], data[7]])


def load_consumers_from_csv():
    """
        Loads consumers from a CSV file and initialises their corresponding Consumer objects.
        If the CSV does not exist, random consumers are generated.

        Returns:
            list of tuples: Each tuple contains (Consumer instance, interests list, associated producer name).
        """
    consumers = []
    if not os.path.exists('consumers.csv'):
        generate_random_consumers(10)
    with open('consumers.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name']
            interests = ast.literal_eval(row['Interests'])
            lag = int(row['Lag'])
            boredom_threshold = int(row['BoredomThreshold'])
            producer_name = row['Producer']
            consumer_type = row.get('Type', 'static').lower()
            is_dynamic = (consumer_type == 'dynamic')
            try:
                novelty_sensitivity = float(row.get('Novelty', 1.0))
            except:
                novelty_sensitivity = 1.0
            reward_function_type = row.get('RewardType', "Fixed")
            consumer = Consumer(name, interests, boredom_threshold=boredom_threshold,
                                cool_off_period=2, preferred_lag=lag, is_dynamic=is_dynamic,
                                resample_interval=10, novelty_sensitivity=novelty_sensitivity,
                                reward_function_type=reward_function_type)
            consumers.append((consumer, interests, producer_name))
            global_consumers[name] = consumer
    return consumers


def load_producers_from_csv():
    """
        Initialises Producer objects from consumer data loaded from a CSV file.
        Assigns sessions to producers based on the producer names specified for each consumer.

        Returns:
            list of Producer: A list of initialised Producer instances.
        """
    consumer_tuples = load_consumers_from_csv()
    producers_dict = {}
    for consumer, interests, producer_name in consumer_tuples:
        session = ConsumerSession(consumer)
        if producer_name not in producers_dict:
            producers_dict[producer_name] = Producer(producer_name)
        producers_dict[producer_name].add_session(session)
    return list(producers_dict.values())


#############################################
# Update Dynamic Consumer CSV Row
#############################################

def update_dynamic_consumer_in_csv(consumer):
    """
    Updates the CSV file for a dynamic consumer to reflect its current interests.

    Parameters:
        consumer (Consumer): The consumer whose data is to be updated.

    """
    csv_file = 'consumers.csv'
    if not os.path.exists(csv_file):
        return
    updated_rows = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['Name'] == consumer.name:
                row['Interests'] = str(list(consumer.true_interests))
            updated_rows.append(row)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)


#############################################
# Consumer Class
#############################################

class Consumer:
    def __init__(self, name, true_interests, boredom_threshold, cool_off_period, preferred_lag,
                 is_dynamic=False, resample_interval=10, novelty_sensitivity=1.0, reward_function_type="Fixed"):
        """
                Initialises a Consumer instance with their properties during the simulation.

                Parameters:
                    name (str): The unique name of the consumer.
                    true_interests (list of int): A binary list representing the consumer's true interests.
                    boredom_threshold (int): The consumer's boredom threshold.
                    cool_off_period (int): The period during which the consumer recovers from boredom and returns.
                    preferred_lag (int): The consumer's preferred lag value.
                    is_dynamic (bool): Indicates whether the consumer dynamically updates their interests.
                    resample_interval (int): The interval (in steps) at which dynamic consumers resample their interests.
                    novelty_sensitivity (float): The consumer's sensitivity or preference for novel content.
                    reward_function_type (str): The preferred type of reward function ("Fixed", "Variable Mean", etc.).

                """
        self.name = name
        self.true_interests = true_interests[:]
        self.boredom_threshold = boredom_threshold
        self.cool_off_period = cool_off_period
        self.preferred_lag = preferred_lag
        self.boredom = 0
        self.click_streak = 0
        self.tau = 5.0  # Time constant for reward decay
        self.last_reward_time = time.time()
        self.is_dynamic = is_dynamic
        self.resample_interval = resample_interval
        self.novelty_sensitivity = novelty_sensitivity
        self.reward_function_type = reward_function_type
        # Flag to indicate an interest change occurred during simulation.
        self.interest_changed = False

        if self.reward_function_type == "Fixed":
            self.mu = 2.0
            self.sigma = 1.0
        elif self.reward_function_type == "Variable Mean":
            self.mu = random.uniform(1, 5)
            self.sigma = 1.0
        elif self.reward_function_type == "Variable SD":
            self.mu = 2.0
            self.sigma = random.uniform(0.1, 2)
        elif self.reward_function_type == "Variable Both":
            self.mu = random.uniform(1, 5)
            self.sigma = random.uniform(0.1, 2)
        else:
            self.mu = 2.0
            self.sigma = 1.0

    def compute_reward_utility(self, current_time, liked_ratio=1.0):
        """
                Computes the reward utility for the consumer based on a Gaussian function, reflecting how
                timely rewards reduce boredom.

                Parameters:
                    current_time (float): The current time (in seconds).
                    liked_ratio (float): The ratio of content that aligns with the consumer's interests.

                Returns:
                    float: The computed reward utility value.
                """
        delta = current_time - self.last_reward_time
        gaussian_component = np.exp(-((delta - self.mu) ** 2) / (2 * self.sigma ** 2))

        return liked_ratio * gaussian_component

    def compute_novelty_utility(self, content, session_knowledge):
        """
                Computes the novelty utility based on the proportion of active content items that are unknown.

                Parameters:
                    content (list of int): A binary list indicating active (1) or inactive (0) content items.
                    session_knowledge (list of str): The consumer's current knowledge state for each content
                    item ("U" for unknown).

                Returns:
                    float: The computed novelty utility value.
                """
        novel_count = sum(1 for i, val in enumerate(content) if val == 1 and session_knowledge[i] == "U")
        total_active = sum(content)
        if total_active == 0:
            return 0
        novelty_ratio = novel_count / total_active
        return self.novelty_sensitivity * novelty_ratio

    def provide_feedback(self, content, current_lag, check_lag=True, session_knowledge=None):
        """
        Determines the consumer's feedback (e.g., "click", "scroll", "leave", "hooked") based on content
        relevance and boredom.

        Parameters:
            content (list of int): The active/inactive state of content items.
            current_lag (int): The current lag value in the simulation.
            check_lag (bool): Flag to enforce a penalty if the lag does not match the consumer's preference.
            session_knowledge (list of str): The consumer's knowledge state for each content item;
            used for novelty computation.

        Returns:
            str: The feedback outcome ("click", "scroll", "leave", or "hooked").
        """
        current_time = time.time()
        liked_count = np.dot(self.true_interests, content)
        total_interest = np.sum(self.true_interests)

        if liked_count == 0:
            self.boredom += 3
            self.click_streak = 0
            if self.boredom >= self.boredom_threshold:
                return "leave"
            return "scroll"
        else:
            pattern_penalty = 1 if (check_lag and current_lag != self.preferred_lag) else 0
            if liked_count == 1:
                increase = 2
            elif liked_count >= 2 and liked_count < total_interest:
                increase = 1
            elif liked_count == total_interest:
                increase = 0

            self.click_streak += 1
            if liked_count == total_interest:
                self.boredom = 0
            else:
                self.boredom += pattern_penalty + increase

            liked_ratio = liked_count / total_interest
            reward_util = self.compute_reward_utility(current_time, liked_ratio)
            novelty_util = 0
            if session_knowledge is not None:
                novelty_util = self.compute_novelty_utility(content, session_knowledge)
            self.boredom = max(0, self.boredom - 0.5 * reward_util - 0.3 * novelty_util)
            self.last_reward_time = current_time

            if self.click_streak >= 10:
                return "hooked"
            if self.click_streak >= 5 and random.random() < 0.1:
                return "leave"
            if self.boredom >= self.boredom_threshold:
                return "leave"
            return "click"

    def resample_preference(self):
        """
        For dynamic consumers: randomly swaps one active interest with one inactive interest and
        updates the CSV record.

        Returns:
            bool: True if an interest change occurred, otherwise False.
        """
        if not self.is_dynamic:
            return False
        active_indices = [i for i, val in enumerate(self.true_interests) if val == 1]
        inactive_indices = [i for i, val in enumerate(self.true_interests) if val == 0]
        if active_indices and inactive_indices:
            off_index = random.choice(active_indices)
            on_index = random.choice(inactive_indices)
            old_off_value = self.true_interests[off_index]
            old_on_value = self.true_interests[on_index]
            self.true_interests[off_index] = 0
            self.true_interests[on_index] = 1
            # Print which consumer resampled which interest.
            print(f"[{self.name}] Resampled interests: index {off_index} changed from {old_off_value} to 0, "
                  f"index {on_index} changed from {old_on_value} to 1")
            self.interest_changed = True
            update_dynamic_consumer_in_csv(self)
            return True
        return False


#############################################
# ConsumerSession Class
#############################################

class ConsumerSession:
    def __init__(self, consumer, num_interests=10, max_story_length=3):
        """
        Initialises a session for a consumer which tracks state such as displayed content and knowledge updates.

        Parameters:
            consumer (Consumer): The consumer associated with this session.
            num_interests (int): The number of interest dimensions.
            max_story_length (int): The maximum number of content items in a story sequence.

        """
        self.consumer = consumer
        self.num_interests = num_interests
        self.max_story_length = max_story_length
        self.knowledge = ["U"] * num_interests
        self.remaining_indices = list(range(num_interests))
        self.current_lag = 0
        self.steps = 0
        self.phase = "discovery"
        self.cycle = 1
        self.active = True

    def generate_content(self, strategy):
        """
        Generates content for the current cycle using the provided strategy.

        Parameters:
            strategy (function): A function that determines which content items should be active.

        Returns:
            numpy.ndarray: An array representing the content state for the current cycle.
        """
        if self.phase == "discovery":
            return strategy(self.steps, self.num_interests, self.knowledge, self.remaining_indices,
                            self.max_story_length)
        else:
            if self.steps % (self.current_lag + 1) == 0:
                return strategy(self.steps, self.num_interests, self.knowledge, self.remaining_indices,
                                self.max_story_length)
            else:
                return np.zeros(self.num_interests)

    def update_knowledge(self, content, feedback):
        """
        Updates the producer's knowledge state based on the consumer's feedback to the presented content.

        Parameters:
            content (list of int): The active/inactive status of content items for the current cycle.
            feedback (str): The consumer feedback.

        """
        active_indices = [i for i, val in enumerate(content) if val == 1]
        if feedback == "click":
            if len(active_indices) == 1:
                i = active_indices[0]
                if self.knowledge[i] in ["U", "M"]:
                    self.knowledge[i] = "I"
            else:
                for i in active_indices:
                    if self.knowledge[i] == "U":
                        self.knowledge[i] = "M"
        elif feedback == "scroll":
            for i in active_indices:
                if self.knowledge[i] in ["U", "M"]:
                    self.knowledge[i] = "N"
            self.remaining_indices = [i for i in self.remaining_indices if i not in active_indices]

    def sync_dynamic_knowledge(self):
        """
        For dynamic consumers, synchronises the session with the consumer's current interests.

        """
        phase_changed = False
        if self.consumer.is_dynamic:
            for i in range(self.num_interests):
                if self.knowledge[i] == "I" and self.consumer.true_interests[i] == 0:
                    self.knowledge[i] = "N"
                    if i not in self.remaining_indices:
                        self.remaining_indices.append(i)
                elif self.knowledge[i] == "N" and self.consumer.true_interests[i] == 1:
                    self.knowledge[i] = "U"
                    if i not in self.remaining_indices:
                        self.remaining_indices.append(i)
                    phase_changed = True
        if phase_changed and self.phase != "discovery":
            self.phase = "discovery"


#############################################
# Producer Class
#############################################

class Producer:
    def __init__(self, name):
        """
        Initialises a Producer instance with a unique name and default rating.
        """
        self.name = name
        self.sessions = {}
        self.rating = 50  # Initial rating set to 50

    def add_session(self, session):
        """
        Creates a consumer session with the associated producer.

        """

        self.sessions[session.consumer.name] = session


#############################################
# Refined Strategy Function
#############################################

def refined_strategy(step, num_interests, knowledge, remaining_indices, max_story_length):
    """
    Determines which topics to activate for the current session step.

    This function returns a binary array of length 'num_interests' where:
      - It first attempts to activate a chunk of "unknown" (U) topics based on the current step and max_story_length.
      - If no U topics are found in that chunk, it activates the first "marginal" (M) topic.
      - If no M topics exist, it activates all "interested" (I) topics.
      - If none of these are found, the feed remains all zeros.

    Parameters:
        step (int): The current simulation step, determining the starting index of the moving window through the list of content items.
        num_interests (int): The total number of topics or interests available.
        knowledge (list of str): A list representing the state of each topic, typically using markers such as "U", "M", or "I".
        remaining_indices (list of int): A list of indices for topics that have not yet been fully explored.
        max_story_length (int): The maximum number of topics to consider in one chunk of content.

    Returns:
        np.ndarray: A binary array of length num_interests with 1s indicating activated topics.
    """

    content = np.zeros(num_interests)

    def activate_indices(idxs):
        arr = np.zeros(num_interests)
        for ix in idxs:
            arr[ix] = 1
        return arr

    if "U" in knowledge:
        chunk_start = (step * max_story_length) % num_interests
        chunk_end = min(chunk_start + max_story_length, num_interests)
        chunk_indices = [i for i in range(chunk_start, chunk_end) if knowledge[i] == "U"]
        if len(chunk_indices) > 0:
            return activate_indices(chunk_indices)
    m_indices = [i for i, status in enumerate(knowledge) if status == "M"]
    if len(m_indices) > 0:
        return activate_indices([m_indices[0]])
    i_indices = [i for i, status in enumerate(knowledge) if status == "I"]
    if len(i_indices) > 0:
        return activate_indices(i_indices)
    return content


#############################################
# Function to Swap Producers for a Consumer
#############################################

def swap_producer(consumer_name, current_producer, new_producer):
    """
        Swaps a consumer from the current producer to a new producer, creating or reactivating a session as necessary.

        Parameters:
            consumer_name (str): The name of the consumer.
            current_producer (Producer): The current producer instance interacting with the consumer.
            new_producer (Producer): The new producer instance to which the consumer is being assigned.

        Returns:
            ConsumerSession: The session associated with the consumer for the new producer.
        """
    current_session = current_producer.sessions.get(consumer_name)
    new_session = new_producer.sessions.get(consumer_name)
    if new_session is not None:
        new_session.active = True
    else:
        consumer = global_consumers.get(consumer_name)
        if consumer is None:
            if current_session is not None:
                consumer = current_session.consumer
            else:
                raise ValueError(f"Consumer {consumer_name} not found.")
        new_session = ConsumerSession(consumer)
        new_producer.add_session(new_session)
    return new_session


#############################################
# Simulation Loop
#############################################

def run_simulation_visual(producers, cool_off_period):
    """
        Runs the main simulation loop, managing producer sessions, generating content, handling consumer feedback,
        and updating the visual state.

        Parameters:
            producers (list of Producer): A list of producer instances participating in the simulation.
            cool_off_period (int): The delay (in seconds) between simulation cycles.

        """
    global simulation_running
    while simulation_running:
        for producer in producers:
            if not simulation_running:
                break
            for session in list(producer.sessions.values()):
                if not session.active:
                    continue
                update_visual_state(producer.name, session.consumer.name,
                                    session.cycle, session.phase, session.current_lag,
                                    session.knowledge, "Starting cycle", session.consumer)
                session.consumer.boredom = 0
                session.consumer.click_streak = 0
                session.steps = 0
                while simulation_running and session.active:
                    content = session.generate_content(refined_strategy)
                    check_lag = (session.phase == "lag")
                    feedback = session.consumer.provide_feedback(content, session.current_lag,
                                                                 check_lag=check_lag,
                                                                 session_knowledge=session.knowledge)
                    session.update_knowledge(content, feedback)
                    update_visual_state(producer.name, session.consumer.name,
                                        session.cycle, session.phase, session.current_lag,
                                        session.knowledge, feedback, session.consumer)
                    if feedback == "click":
                        producer.rating = min(95, producer.rating + 5)
                    elif feedback == "leave":
                        producer.rating = max(5, producer.rating - 5)
                        update_visual_state(producer.name, session.consumer.name,
                                            session.cycle, session.phase, session.current_lag,
                                            session.knowledge, "left due to boredom", session.consumer)
                        session.active = False
                        other_producers = [p for p in producers if p != producer]
                        if other_producers:
                            new_producer = choose_producer_weighted(other_producers)
                            new_session = swap_producer(session.consumer.name, producer, new_producer)
                            if (new_producer.name, session.consumer.name) not in visual_frames:
                                vsf = VisualSessionFrame(producer_frames[new_producer.name],
                                                         new_producer.name, session.consumer.name)
                                vsf.pack(pady=5, fill=tk.X)
                                visual_frames[(new_producer.name, session.consumer.name)] = vsf
                        break
                    elif feedback == "hooked":
                        update_visual_state(producer.name, session.consumer.name,
                                            session.cycle, session.phase, session.current_lag,
                                            session.knowledge, "hooked", session.consumer)
                        session.active = False
                        break
                    session.steps += 1
                    # For dynamic consumers, check if it's time to resample interests.
                    if session.consumer.is_dynamic and (session.steps % session.consumer.resample_interval == 0):
                        changed = session.consumer.resample_preference()
                        session.sync_dynamic_knowledge()
                    time.sleep(0.5)
                if session.active:
                    if session.phase == "discovery":
                        if all(k in ["I", "N"] for k in session.knowledge):
                            session.phase = "lag"
                            update_visual_state(producer.name, session.consumer.name,
                                                session.cycle, session.phase, session.current_lag,
                                                session.knowledge, "All topics discovered", session.consumer)
                    else:
                        session.current_lag += 1
                        update_visual_state(producer.name, session.consumer.name,
                                            session.cycle, session.phase, session.current_lag,
                                            session.knowledge, "Adjusting lag", session.consumer)
                    time.sleep(cool_off_period)
                    session.cycle += 1
        active_sessions = any(session.active for prod in producers for session in prod.sessions.values())
        if not active_sessions:
            simulation_running = False
            dummy_consumer = type("DummyConsumer", (),
                                  {"boredom_threshold": "N/A", "boredom": "N/A", "true_interests": [],
                                   "is_dynamic": False, "reward_function_type": "N/A", "interest_changed": False})()
            update_visual_state("GLOBAL", "ALL", 0, "STOPPED", 0, [], "All consumers hooked", dummy_consumer)
    dummy_consumer = type("DummyConsumer", (), {"boredom_threshold": "N/A", "boredom": "N/A", "true_interests": [],
                                                "is_dynamic": False, "reward_function_type": "N/A",
                                                "interest_changed": False})()
    update_visual_state("GLOBAL", "ALL", 0, "STOPPED", 0, [], "Simulation stopped", dummy_consumer)


#############################################
# Visual Session Frame (Tkinter)
#############################################

class VisualSessionFrame(tk.Frame):
    COLORS = {
        "I": "green",
        "N": "red",
        "M": "yellow",
        "U": "gray"
    }

    def __init__(self, master, producer_name, consumer_name):
        """
        Initialises a VisualSessionFrame for displaying session details in the GUI.

        Parameters:
            master (Tk widget): The parent widget.
            producer_name (str): The name of the producer associated with this session.
            consumer_name (str): The name of the consumer in the session.

        """
        super().__init__(master, bd=2, relief=tk.RIDGE, padx=5, pady=5)
        self.producer_name = producer_name
        self.consumer_name = consumer_name
        header_text = f"{producer_name} - {consumer_name}"
        self.header_label = tk.Label(self, text=header_text, font=("Arial", 12, "bold"))
        self.header_label.pack()
        self.info_label = tk.Label(self, text="Cycle: -, Phase: -, Lag: -", font=("Arial", 10))
        self.info_label.pack()
        self.feedback_label = tk.Label(self, text="Feedback: -", font=("Arial", 10))
        self.feedback_label.pack()
        self.canvas = tk.Canvas(self, width=300, height=30, bg="white")
        self.canvas.pack(pady=5)
        self.rects = []
        for i in range(10):
            rect = self.canvas.create_rectangle(5 + i * 28, 5, 5 + i * 28 + 25, 25, fill="gray")
            self.rects.append(rect)

    def update_state(self, cycle, phase, current_lag, knowledge, feedback):
        """
        Updates the visual elements of the frame to reflect the current simulation state.

        Parameters:
            cycle (int): The current simulation cycle.
            phase (str): The current phase of the session.
            current_lag (int): The current lag value.
            knowledge (list of str): The updated knowledge state for the session.
            feedback (str): The latest feedback received.

        """
        self.info_label.config(text=f"Cycle: {cycle}, Phase: {phase}, Lag: {current_lag}")
        self.feedback_label.config(text=f"Feedback: {feedback}")
        for i, state in enumerate(knowledge):
            color = self.COLORS.get(state, "gray")
            self.canvas.itemconfig(self.rects[i], fill=color)


#############################################
# GUI Setup with Scrollable Sessions Frame in Two Columns
#############################################

root = tk.Tk()
root.title("Visual Multi-Producer Simulation")

container = tk.Frame(root)
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

sessions_container = tk.Frame(canvas)
canvas.create_window((0, 0), window=sessions_container, anchor="nw")


def on_frame_configure(event):
    """
    Configures the canvas scroll region based on the frame's size whenever a configuration event occurs.

    Parameters:
        event: The event object with details of the configuration change.

    """
    canvas.configure(scrollregion=canvas.bbox("all"))


sessions_container.bind("<Configure>", on_frame_configure)

producer_frames = {}
producers = load_producers_from_csv()
for producer in producers:
    frame = tk.Frame(sessions_container, bd=2, relief=tk.GROOVE)
    frame.pack(side=tk.LEFT, fill="both", expand=True, padx=10, pady=10)
    header = tk.Label(frame, text=producer.name, font=("Arial", 14, "bold"))
    header.pack(pady=5)
    producer_frames[producer.name] = frame

visual_frames = {}
for producer in producers:
    for consumer_name, session in producer.sessions.items():
        vsf = VisualSessionFrame(producer_frames[producer.name], producer.name, consumer_name)
        vsf.pack(pady=5, fill=tk.X)
        visual_frames[(producer.name, consumer_name)] = vsf


def poll_visual_state():
    """
        Periodically polls the shared visual state and updates the GUI accordingly.

        """

    with state_lock:
        for key, state in visual_state.items():
            if key in visual_frames:
                frame = visual_frames[key]
                frame.update_state(state["cycle"], state["phase"], state["current_lag"],
                                   state["knowledge"], state["latest_feedback"])
    root.after(100, poll_visual_state)


#############################################
# Start/Stop Simulation Functions
#############################################

def start_simulation_button():
    """
        Initiates the simulation by starting the simulation thread.

        """
    global simulation_running
    if not simulation_running:
        simulation_running = True
        sim_thread = threading.Thread(target=run_simulation_visual, args=(producers, 2), daemon=True)
        sim_thread.start()


def stop_simulation_button():
    """
        Stops the simulation by setting the control flag to False.

        """
    global simulation_running
    simulation_running = False


#############################################
# Buttons
#############################################

button_frame = tk.Frame(root)
button_frame.pack(pady=10)
start_button = tk.Button(button_frame, text="Start Simulation", command=start_simulation_button)
start_button.pack(side=tk.LEFT, padx=5)
stop_button = tk.Button(button_frame, text="Stop Simulation", command=stop_simulation_button)
stop_button.pack(side=tk.LEFT, padx=5)

poll_visual_state()
root.mainloop()
