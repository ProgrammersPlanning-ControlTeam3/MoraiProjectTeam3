#!/usr/bin/env python3

import json

class ScenarioLoader():
    def __init__(self):
        self.path = '/home/nodazi24/Downloads/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Scenario/R_KR_PG_K-City/random_obstacle_scenario.json'
        with open(self.path, 'r') as f:
            self.data = json.load(f)
        self.objectList = self.data['objectList']

    def getObject(self) :
        return self.objectList
