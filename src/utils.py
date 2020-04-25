class AvgMeter():
    def __init__(self):
        self.val = 0.0
        self.cnt = 0

    def update(self, v, c):
        self.val += v
        self.cnt += c
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else:
            return self.val/self.cnt