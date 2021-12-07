import numpy as np
def Normalize(data):
     m = np.mean(data)
     mx = max(data)
     mn = min(data)
     return [(float(i) - m) / (mx - mn) for i in data]