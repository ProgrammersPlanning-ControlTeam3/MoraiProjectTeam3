#!/usr/bin/env python3

import json
import random


def random_boolean(probability):
    return random.random() < probability

def main():
    # base scenario file
    base_file_path = 'Morai_Project/scenario/random_obstacle_base.json'

    with open(base_file_path, 'r') as f:
        data = json.load(f)

    object_lists = data['objectList']
    blankParkingSpace = []

    n = 0
    numParkingVehicles = 30
    while len(object_lists) > numParkingVehicles:
        if object_lists[n]['DataID'] == 40100019:
            if random_boolean(0.3):
                obj = object_lists.pop(n)
                blankParkingSpace.append(obj['pos'])
                # print(obj['pos'])
        n += 1
        if n >= len(object_lists):
            n = 0

    with open('Morai_Project/scenario/random_obstacle_scenario.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(random.choice(blankParkingSpace))


if __name__ == "__main__":
    main()
