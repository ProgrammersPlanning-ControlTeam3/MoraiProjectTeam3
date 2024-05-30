#!/usr/bin/env python3

import json
import random


def random_boolean(probability):
    return random.random() < probability

def main():
    # base scenario file
    base_file_path = 'src/random_scenario_maker/scenario/random_obstacle_base.json'

    with open(base_file_path, 'r') as f:
        data = json.load(f)

    object_lists = data['objectList']

    parking_lot_start = data['parkingLotStart']
    parking_lot_goal = data['parkingLotGoal']
    
    blankParkingSpace = []

    cnt = 0
    numParkingVehicles = 30
    while len(object_lists) > numParkingVehicles:
        if object_lists[cnt]['DataID'] == 40100019:
            if random_boolean(0.3):
                obj = object_lists.pop(cnt)
                pos = obj['pos']
                rot = obj['rot']
                blankParkingSpace.append((pos,rot))
                # print(obj['pos'])
        cnt += 1
        if cnt >= len(object_lists):
            cnt = 0
    
    #fixed start point
    parking_lot_start['pos'] = {
        "x": 64.850997924804688,
        "y": 1851.362060546875,
        "z": -0.32499998807907104,
        "_x": "64.851",
        "_y": "1,851.362",
        "_z": "-0.325"
      }
    parking_lot_start['rot'] = {
        "roll": "0.0",
        "pitch": "0.0",
        "yaw": "-35.765"
        }

    random_goal = random.choice(blankParkingSpace)
    parking_lot_goal['pos'] = random_goal[0]
    parking_lot_goal['rot'] = random_goal[1]

    with open('src/random_scenario_maker/scenario/random_obstacle_scenario.json', 'w') as f:
        json.dump(data, f, indent=2)

    # print(random.choice(blankParkingSpace))


if __name__ == "__main__":
    main()
