import numpy as np

class Rectangle:
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def _triangle_area(self, p1, p2, p3):
        return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)

    def is_inside(self, point):
        # 사각형의 네 꼭지점과 주어진 점이 이루는 네 삼각형의 면적 합을 계산
        total_area = self._triangle_area(self.p1, self.p2, self.p3) + \
                     self._triangle_area(self.p1, self.p3, self.p4)
        
        area1 = self._triangle_area(point, self.p1, self.p2)
        area2 = self._triangle_area(point, self.p2, self.p3)
        area3 = self._triangle_area(point, self.p3, self.p4)
        area4 = self._triangle_area(point, self.p4, self.p1)
        # print(total_area)
        # print(area1 + area2 + area3 + area4)
        
        # 사각형의 총 면적과 점을 포함한 네 삼각형의 면적 합을 비교
        return total_area == (area1 + area2 + area3 + area4)