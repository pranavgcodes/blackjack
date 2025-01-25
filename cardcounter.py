from typing import List, Dict, Union, Tuple, Optional
from collections import defaultdict
import json
import logging
import threading

########################################################
# 1. Card Model
########################################################

class Card:
    """
    Represents a playing card with a rank and suit.
    Rank: '2','3','4','5','6','7','8','9','10','J','Q','K','A'
    Suit: '♠','♥','♦','♣'
    """
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def value(self) -> int:
        """
        For Blackjack:
        - 2-10 => numeric
        - J/Q/K => 10
        - A => 1 (the '11' logic is handled elsewhere)
        """
        if self.rank in ['J','Q','K','10']:
            return 10
        elif self.rank == 'A':
            return 1
        else:
            return int(self.rank)

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

    def __hash__(self):
        return hash((self.rank, self.suit))

########################################################
# 2. CardCounter
########################################################

class CardCounter:
    """
    Maintains a dictionary of all 'unseen' cards for a given number of decks.
    Observing a card decrements its count in unseen_cards and updates it:
      - Running count based on which method
      - Side count of Aces
    """
    def __init__(self, 
                 num_decks: int = 6, 
                 counting_system: str = 'Hi-Lo',
                 track_aces_side_count: bool = False):
        self.num_decks = num_decks
        self.counting_system = counting_system
        self.track_aces_side_count = track_aces_side_count

        self.unseen_cards = self._build_full_shoe()
        self.running_count = 0
        self.aces_count = 0 

        # Lock for thread safety
        self.lock = threading.Lock()

    def _build_full_shoe(self) -> Dict[Card, int]:
        ranks = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
        suits = ['♠','♥','♦','♣']
        card_counts = defaultdict(int)
        for _ in range(self.num_decks):
            for r in ranks:
                for s in suits:
                    card_counts[Card(r, s)] += 1
        return card_counts

    def observe_card(self, rank: str, suit: str):
        """
        Whenever your CV model spots a new card, call this.
        rank: '2','3','4','5','6','7','8','9','10','J','Q','K','A'
        suit: '♠','♥','♦','♣'
        """
        card_observed = Card(rank, suit)

        with self.lock:
            if self.unseen_cards[card_observed] > 0:
                self.unseen_cards[card_observed] -= 1
                if self.unseen_cards[card_observed] == 0:
                    del self.unseen_cards[card_observed]
                self._update_running_count(card_observed)
            else:
                logging.warning(f"Attempted to observe card {rank}{suit}, but it was not found in unseen_cards.")

    def _update_running_count(self, card: Card):
        rank = card.rank

        # Side counting for Aces
        if self.track_aces_side_count and rank == 'A':
            self.aces_count += 1

        # HI-LO
        if self.counting_system == 'Hi-Lo':
            if rank in ['2','3','4','5','6']:
                self.running_count += 1
            elif rank in ['10','J','Q','K','A']:
                self.running_count -= 1

        # OMEGA II
        elif self.counting_system == 'Omega II':
            # 2,3,7 => +1
            # 4,5,6 => +2
            # 9 => -1
            # 10,J,Q,K => -2
            # A => 0
            if rank in ['2','3','7']:
                self.running_count += 1
            elif rank in ['4','5','6']:
                self.running_count += 2
            elif rank == '9':
                self.running_count -= 1
            elif rank in ['10','J','Q','K']:
                self.running_count -= 2

        # ZEN
        elif self.counting_system == 'Zen':
            # 2,3,7 => +1
            # 4,5,6 => +2
            # 10,J,Q,K => -2
            # A => -1
            # 8,9 => 0
            if rank in ['2','3','7']:
                self.running_count += 1
            elif rank in ['4','5','6']:
                self.running_count += 2
            elif rank in ['10','J','Q','K']:
                self.running_count -= 2
            elif rank == 'A':
                self.running_count -= 1

    def cards_remaining(self) -> int:
        with self.lock:
            return sum(self.unseen_cards.values())

    def decks_remaining_estimate(self) -> float:
        return self.cards_remaining() / 52.0

    def true_count(self) -> float:
        decks_remaining = self.decks_remaining_estimate()
        if decks_remaining <= 0:
            logging.warning("No decks remaining. True count is undefined.")
            return 0.0
        return self.running_count / decks_remaining

    def get_info_dict(self) -> Dict:
        """
        Returns a dict summarizing the counting state.
        """
        return {
            "counting_system": self.counting_system,
            "running_count": self.running_count,
            "true_count": self.true_count(),
            "decks_remaining_estimate": self.decks_remaining_estimate(),
            "aces_side_count": self.aces_count if self.track_aces_side_count else None,
            "cards_unseen": self.cards_remaining()
        }

########################################################
# 3. Hand Value & Utility
########################################################

def calculate_hand_value(cards: List[Card]) -> int:
    """
    Typical Blackjack hand value with Aces as 1 or 11.
    """
    total = 0
    aces = 0
    for c in cards:
        total += c.value()
        if c.rank == 'A':
            aces += 1

    while aces > 0:
        if total + 10 <= 21:
            total += 10
        aces -= 1

    return total


def is_soft_hand(cards: List[Card]) -> bool:
    """
    Returns True if the hand includes at least one Ace counted as 11.
    """
    total = calculate_hand_value(cards)
    aces = sum(1 for c in cards if c.rank == 'A')
    return aces > 0 and (total - aces + 11 <= 21)


def _hand_key_for_pairs(cards: List[Card]) -> Union[str, int]:
    """
    Returns either:
    - A string like '10,10' or '8,8' if exactly 2 cards of the same rank (pairs)
    - Otherwise the integer total for the hand (e.g., 16)
    """
    if len(cards) == 2 and cards[0].rank == cards[1].rank:
        # e.g., '10,10' or '8,8' or '7,7'
        return f"{cards[0].rank},{cards[1].rank}"
    else:
        return calculate_hand_value(cards)

########################################################
# 4. Top 50 Deviations Implementation
########################################################

"""
Structure: 
  - The player's scenario (like '16 vs 10')
  - The threshold 
  - The action ("Stand", "Surrender", "Split", "Double", "Insure", etc.)

We convert "10" -> 10, 
We convert "Stand"->'S', "Surrender"->'R', "Split"->'P', "Double"->'D', "Insure" is a special case (handled below).
"""

# List of 50 deviations
MOST_IMPORTANT_DEVIATIONS = [
    ["Insurance", 3, "Insure"],
    ["16 vs T", 0, "Stand"],
    ["14 vs T", 3, "Surrender"],
    ["15 vs T", 1, "Stand"],
    ["T,T vs 6", 3, "Split"],
    ["T,T vs 5", 3, "Split"],
    ["15 vs 2", 2, "Stand"],
    ["15 vs T", 4, "Surrender"],
    ["15 vs T", 1, "Surrender"],
    ["15 vs 4", 1, "Stand"],
    ["15 vs 2", 7, "Stand"],
    ["T,T vs 4", 7, "Split"],
    ["8,8 vs A", 5, "Surrender"],
    ["10 vs 9", 3, "Double"],
    ["16 vs 8", 4, "Surrender"],
    ["16 vs 8", 3, "Surrender"],
    ["16 vs 7", 3, "Double"],
    ["18 vs 6", 3, "Double"],
    ["19 vs 7", 4, "Double"],
    ["19 vs 7", 4, "Stand"],
    ["17 vs <2", 2, "Surrender"],
    ["A,8 vs 5", 1, "Double"],
    ["7,7 vs 3", 2, "Surrender"],
    ["14 vs 4", 2, "Surrender"],
    ["10 vs 7", 4, "Double"],
    ["13 vs 2", -1, "Stand"],
    ["14 vs 9", 7, "Surrender"],
    ["15 vs 3", 1, "Surrender"],
    ["T,T vs 3", 7, "Split"],
    ["16 vs 9", -5, "Stand"],
    ["16 vs T", -3, "Surrender"],
    ["10 vs 7", 7, "Double"],
    ["15 vs A", 7, "Surrender"],
    ["12 vs 5", -2, "Stand"],
    ["11 vs 4", 3, "Double"],
    ["16 vs 9", -1, "Surrender"],
    ["A,9 vs 4", -4, "Double"],
    ["13 vs 3", -3, "Double"],
    ["A,8 vs 3", 3, "Double"],
    ["A,9 vs 5", 5, "Double"],
    ["15 vs 5", -1, "Surrender"],
    ["16 vs 9", -3, "Stand"],
    ["16 vs 9", -2, "Stand"],
    ["T,T vs 2", -11, "Split"],
    ["12 vs 6", -5, "Stand"],
    ["13 vs 4", -4, "Stand"],
    ["13 vs 5", -4, "Stand"],
    ["14 vs 5", -4, "Double"],
    ["A,3 vs 4", 2, "Double"]
]

def _parse_scenario(row: List) -> Optional[Tuple[Tuple[Union[str, int], Union[int, str]], Tuple[int, str]]]:
    """
    Turns an entry like ["16 vs 10", 0, "Stand"] 
    into a dict key like (16, 10) => (threshold=0, alt_action='S').

    For pairs like "10,10 vs 6", we do ("10,10", 6).
    For 'Surrender' we map it to 'R', 'Stand'->'S', 'Double'->'D', 'Split'->'P'.
    If it's "Insurance", we skip it here and handle separately.
    """
    scenario_str, threshold, action_str = row[0], row[1], row[2]

    # "Insurance" is a special case => skip here
    if scenario_str.lower().startswith("insurance"):
        return None

    # Map the action string
    action_map = {
        "Stand": "S",
        "Surrender": "R",
        "Split": "P",
        "Double": "D"
    }
    alt_action = action_map.get(action_str, None)
    if alt_action is None:
        return None

    try:
        left_side, right_side = scenario_str.split(" vs ")
    except ValueError:
        return None

    # Dealer side
    if right_side == 'T':
        dealer_value = 10
    elif right_side == 'A':
        dealer_value = 1
    elif right_side.startswith('<'):  
        # e.g., "17 vs <2" => interpret as "vs 2"
        # Assuming "<2" implies "vs 2"
        try:
            dealer_value = int(right_side[1:])  # Extract the number after '<'
        except ValueError:
            return None
    else:
        try:
            dealer_value = int(right_side)
        except ValueError:
            # Handle cases like "vs T" if any remain
            if right_side == 'T':
                dealer_value = 10
            else:
                return None

    # Player side
    # If it looks like "10,10" or "8,8", handle as a pair key
    if ',' in left_side:
        # e.g., "10,10"
        # We'll keep it exactly as "10,10"
        # Then in code, if the player's 2-card ranks == 10,10, we match it
        player_key = left_side.replace('T', '10')  # Standardize 'T' to '10'
    else:
        # single total or something like "14"
        if left_side == 'T':
            player_key = 10
        elif left_side == 'A':
            player_key = 1
        else:
            try:
                player_key = int(left_side)
            except ValueError:
                # Handle cases like 'A,8' if they are not treated as pairs
                player_key = left_side  # Keep as string if not a pure number

    # Standardize 'T' to '10' in player_key if it's a string
    if isinstance(player_key, str):
        player_key = player_key.replace('T', '10')

    return ((player_key, dealer_value), (threshold, alt_action))

# Build the dictionary from the user-provided table
# Mapping: (player_key, dealer_val) -> List of (threshold, alt_action)
UPDATED_INDEX_DEVIATIONS: Dict[Tuple[Union[str, int], Union[int, str]], List[Tuple[int, str]]] = defaultdict(list)
for row in MOST_IMPORTANT_DEVIATIONS:
    parsed = _parse_scenario(row)
    if parsed is not None:
        (key, val) = parsed
        UPDATED_INDEX_DEVIATIONS[key].append(val)

# Sort the deviations for each scenario based on thresholds
# For positive thresholds: sort descending
# For negative thresholds: sort ascending
for key in UPDATED_INDEX_DEVIATIONS:
    deviations = UPDATED_INDEX_DEVIATIONS[key]
    # Separate positive and negative thresholds
    pos_devs = sorted([d for d in deviations if d[0] >= 0], key=lambda x: x[0], reverse=True)
    neg_devs = sorted([d for d in deviations if d[0] < 0], key=lambda x: x[0])
    # Combine them: positive thresholds first (higher to lower), then negative (lower to higher)
    UPDATED_INDEX_DEVIATIONS[key] = pos_devs + neg_devs

########################################################
# 5. Index-Based Deviation Function
########################################################

def revised_index_deviation(player_cards: List[Card],
                            dealer_upcard: Card,
                            true_count: float) -> Optional[str]:
    """
    Checks if there's an index-based deviation from your new table.
    Returns 'S','H','D','P','R' if a deviation applies, else None.

    1) If the player has exactly 2 same-rank cards, we look up ("10,10", dealerVal)
    2) Otherwise, we look up (total, dealerVal)
    
    Handles both positive and negative thresholds.
    """
    # Check for pairs, e.g., "10,10" or "7,7"
    pair_key = _hand_key_for_pairs(player_cards)
    dealer_val = dealer_upcard.value()

    scenario_key = (pair_key, dealer_val)

    if scenario_key in UPDATED_INDEX_DEVIATIONS:
        deviations = UPDATED_INDEX_DEVIATIONS[scenario_key]
        for threshold, alt_action in deviations:
            if threshold >= 0:
                if true_count >= threshold:
                    return alt_action
            else:
                if true_count <= threshold:
                    return alt_action
    return None

########################################################
# 6. Handling of Insurance
########################################################

def recommend_insurance(dealer_upcard: Card, card_counter: CardCounter) -> bool:
    """
    Among your lines is ["Insurance", 3, "Insure", ...].
    This means if the dealer's upcard is an Ace and 
    true_count >= 3 => take insurance.
    """
    if dealer_upcard.rank == 'A':
        if card_counter.true_count() >= 3:
            return True
    return False

########################################################
# 7. Basic Strategy Integration
########################################################

BASIC_STRATEGY = {
    "hard": {
        "8": "H",
        "9": {"3": "D", "4": "D", "5": "D", "6": "D", "default": "H"},
        "10": {"2": "D", "3": "D", "4": "D", "5": "D", "6": "D", "7": "D", "8": "D", "9": "D", "default": "H"},
        "11": {"default": "D"},
        "12": {"4": "S", "5": "S", "6": "S", "default": "H"},
        "13": {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S", "default": "H"},
        "14": {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S", "default": "H"},
        "15": {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S", "default": "H"},
        "16": {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S", "default": "H"},
        "17": "S",
        "18": "S",
        "19": "S",
        "20": "S",
        "21": "S"
    },
    "soft": {
        "13": {"5": "D", "6": "D", "default": "H"},
        "14": {"5": "D", "6": "D", "default": "H"},
        "15": {"4": "D", "5": "D", "6": "D", "default": "H"},
        "16": {"4": "D", "5": "D", "6": "D", "default": "H"},
        "17": {"3": "D", "4": "D", "5": "D", "6": "D", "default": "H"},
        "18": {"3": "D", "4": "D", "5": "D", "6": "D", "default": "S"},
        "19": {"6": "D", "default": "S"},
        "20": "S",
        "21": "S"
    },
    "pairs": {
        "A,A": "P",
        "10,10": "S",
        "9,9": {"2": "P", "3": "P", "4": "P", "5": "P", "6": "P", "8": "P", "9": "S", "7": "S", "default": "S"},
        "8,8": "P",
        "7,7": {"2": "P", "3": "P", "4": "P", "5": "P", "6": "P", "7": "P", "default": "H"},
        "6,6": {"2": "P", "3": "P", "4": "P", "5": "P", "6": "P", "default": "H"},
        "5,5": {"default": "D"},
        "4,4": {"5": "P", "6": "P", "default": "H"},
        "3,3": {"2": "P", "3": "P", "4": "P", "5": "P", "6": "P", "7": "P", "default": "H"},
        "2,2": {"2": "P", "3": "P", "4": "P", "5": "P", "6": "P", "7": "P", "default": "H"}
    }
}

def load_basic_strategy(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def get_basic_strategy_action(player_cards: List[Card], dealer_upcard: Card) -> str:
    player_total = calculate_hand_value(player_cards)
    soft = is_soft_hand(player_cards)
    pair_key = _hand_key_for_pairs(player_cards)
    dealer_val_str = str(dealer_upcard.value())

    # Check for pairs
    if isinstance(pair_key, str) and pair_key in BASIC_STRATEGY["pairs"]:
        action = BASIC_STRATEGY["pairs"][pair_key]
        if isinstance(action, dict):
            return action.get(dealer_val_str, action.get("default", "H"))
        else:
            return action

    # Check for soft hands
    if soft and str(player_total) in BASIC_STRATEGY["soft"]:
        actions = BASIC_STRATEGY["soft"][str(player_total)]
        if isinstance(actions, dict):
            return actions.get(dealer_val_str, actions.get("default", "H"))
        else:
            return actions

    # Check for hard hands
    if not soft and str(player_total) in BASIC_STRATEGY["hard"]:
        actions = BASIC_STRATEGY["hard"][str(player_total)]
        if isinstance(actions, dict):
            return actions.get(dealer_val_str, actions.get("default", "H"))
        else:
            return actions

    # Default action
    return 'H'

########################################################
# 8. Action Description Mapping
########################################################

ACTION_DESCRIPTIONS = {
    'H': 'Hit',
    'S': 'Stand',
    'D': 'Double',
    'P': 'Split',
    'R': 'Surrender',
    'I': 'Insure'
}

########################################################
# 9. Action Recommendation Function
########################################################

def recommend_action_for_player_hand(
    player_cards: List[Card],
    dealer_upcard: Card,
    card_counter: CardCounter,
    house_rules: Dict = None
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns a tuple containing:
    - 'I' if insurance is recommended, else None
    - Primary action: 'H','S','D','P','R'
    - Description of the action
    """
    if house_rules is None:
        house_rules = {
            'hit_soft_17': True,
            'late_surrender': True,
            'double_after_split': True
        }

    # 1) Check for insurance first
    insurance = 'I' if recommend_insurance(dealer_upcard, card_counter) else None

    # 2) Check for index-based deviation from your table
    dev_action = revised_index_deviation(
        player_cards,
        dealer_upcard,
        card_counter.true_count()
    )
    if dev_action is not None:
        return (insurance, dev_action, ACTION_DESCRIPTIONS.get(dev_action, 'Unknown Action'))

    # 3) If no deviation, use basic strategy
    basic_action = get_basic_strategy_action(player_cards, dealer_upcard)
    return (insurance, basic_action, ACTION_DESCRIPTIONS.get(basic_action, 'Unknown Action'))

