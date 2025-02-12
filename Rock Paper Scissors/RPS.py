# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random

def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    # Strategy to counter different opponents
    if len(opponent_history) == 0:
        return random.choice(["R","P","S"]) # First move is random

    # Quincy : Fixed pattern ["R","R","P","P","S"]
    quincy_pattern = ["R","R","P","P","S"]
    if len(opponent_history) >= 5 and opponent_history[-5:] == quincy_pattern:
        return "S"
    
    # Kris : Always counters the last move we played
    if len(opponent_history) >= 1:
        last_move = opponent_history[-1]
        counter_move = {"R":"P","P":"S","S":"R"} # Always counter Last move
        return counter_move[last_move]

    # Mrugesh : Plays based on the most frequent move in the last 10 rounds
    
    if len(opponent_history) >= 10:
        last_ten = opponent_history[-10:]
        most_frequent = max(set(last_ten),key=last_ten.count)
        ideal_response = {"P":"S","R":"P","S":"R"}
        return ideal_response[most_frequent] 
    
    # Abbey: Uses bigram frequency tracking   
    if len(opponent_history) >= 2:
        last_two = "".join(opponent_history[-2:])
        play_order = {
            "RR":"P","RP":"S","RS":"R",
            "PR":"S","PP":"R","PS":"P",
            "SR":"R","SP":"P","SS":"S"
        }
        return play_order.get(last_two,random.choice(["R","P","S"]))

    return random.choice(["R","P","S"]) # Default fallback move

