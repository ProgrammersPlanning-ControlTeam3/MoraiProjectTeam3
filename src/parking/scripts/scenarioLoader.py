#!/usr/bin/env python3

import json

"""
모라이 시뮬레이터 Scenario의 경로에서
시나리오 파일을 불러옴.
"""
class ScenarioLoader():
    def __init__(self):
        ## Morai 경로
        self.path = '/home/nodazi24/Downloads/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Scenario/R_KR_PG_K-City/test.json'

        #Random 시나리오
        # self.path = 'src/random_scenario_maker/scenario/random_obstacle_scenario_for_test.json'
        #Sample 시나리오
        # self.path = '/home/nodazi24/morai_final_second_ws/MoraiProjectTeam3/src/random_scenario_maker/scenario/random_obstacle_scenario.json'
        
        with open(self.path, 'r') as f:
            self.data = json.load(f)
        self.objectList = self.data['objectList']
        self.parkingLotStart = self.data['parkingLotStart']
        self.parkingLotGoal = self.data['parkingLotGoal']
    def getObject(self) :
        return self.objectList

    def getParkingLotStart(self) :
        return self.parkingLotStart

    def getParkingLotGoal(self) :
        return self.parkingLotGoal
