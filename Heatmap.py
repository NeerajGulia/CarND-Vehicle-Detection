import numpy as np
class Heatmap:    
    def __init__(self, threshold):
        self.__heatMapList = []
        self.__threshold = threshold
     
    def pushIn(self, heatmap):
        if len(self.__heatMapList) >= self.__threshold:
            self.__pushOut()
        self.__heatMapList.append(heatmap)
        return self.pullAvg()
            
    def __pushOut(self):
        self.__heatMapList = self.__heatMapList[1:]
        
    def pullAvg(self):
        if len(self.__heatMapList) == 0:
            return ([])
        else:
            return( np.mean(self.__heatMapList, axis = 0))
