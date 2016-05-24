
# coding: utf-8

# In[1104]:

from random import randint, uniform
import matplotlib.pyplot as plt
import sys

def main():
    plt.figure(1)
    #numClusters = 10
    #fileName = "file.txt"
    
    if (len(sys.argv) != 3):
        print("Please run program as follows: Clustering.py numberOfClusters filename")
        exit(1)
        
    numClusters = int(sys.argv[1])
    fileName = sys.argv[2]
    
    
    kmeans(numClusters, fileName)
    
    


# In[1105]:

def kmeans(numClusters, fileName):
    
    #points: [((x, y), cluster), (point, cluster), ...]
    points = loadFile(fileName)
        
    # seeds: [((x, y), cluster), (point, cluster), ...]
    seeds = []
    #red is cluster 0
    colors = [(1, 0, 0)]
    
    # any point in seeds is not found in points. seeds initially come from points.
    # num points = num original points - numClusters
    
    if (numClusters > len(points)):
        numClusters = len(points)
        print("Please do not make numClusters greater than the number of unique points. k is now: " + str(numClusters))
    
    #inits random colors to use for the classifications
    for k in range(0, numClusters):
        colors.append((uniform(0, 1), uniform(0, 1), uniform(0, 1)))
    
    #initializes all the points to have default cluster: 0
    #i = 0
    #for i in range(0, len(points)):
    #    pointClusters.append(0)
    
    
    seeds = [[]]
    pointsTuples, pointsCluster = zip(*points)
    #creates seeds
    for i in range(1, numClusters + 1):
        
        seedsPoints = []
        seedsClusters = []
        if (len(seeds) > 1):
            seedsPoints, seedsClusters = zip(*seeds[1:])
        
        n = 0
        index = randint(0, len(points) - 1)
        
        while(pointsTuples[index] in seedsPoints and n < 10):
            n += 1
            index = randint(0, len(points) - 1)                   
        
        seed = pointsTuples[index]
        seeds.append([seed, i])
        #pointClusters.pop[index]
        
        # seed already established here, find a new one
        #while (pointClusters[index] != 0 and n < 10):
        #    index = randint(0, len(points) - 1)
        #    n += 1
            
        #seed = points[index]
        #pointClusters[index] = i
        #seeds.append([seed, index])
        
    #print(str(seeds))
        
    plt.subplot(2, 2, 1)
    plt.axis([0, 20, 0, 20])
    
    
    pointsList, pointsClusters = zip(*points)
    pointsX, pointsY = zip(*pointsList)
    seedPoints, clusters = zip(*seeds[1:])
    seedsX, seedsY = zip(*seedPoints)
    
    for i in range(0, len(points)):
        #print("SEEDS: " + str(seedPoints) + ", POINT: " + str(pointsList[i]))
        if (pointsList[i] not in seedPoints):
            curCluster = points[i][1]
            c = colors[curCluster]
        
            plt.plot(pointsX[i], pointsY[i], color=c, marker='o')
        
        
    for i in range(1, len(seeds)):
        #print(seeds[i][1])
        curCluster = seeds[i][1]
        c = colors[curCluster]        
        
        plt.plot(seedsX[i-1], seedsY[i-1], color=c, marker='*')
        
    isDone = False
    graph = 2
    figure = 1
    
    #print("Starting seeds: " + str(seeds))    
    #print("Starting points: " + str(points))
    
    # timeout after 100 iterations, probably didn't converge
    timeout = 100
    while((isDone == False) and (timeout > 0)):
        
        isDone = True
        for i in range(0, len(points)):

            minDist = sys.maxsize
            minClusters = []

            for seed in seeds:
                if (len(seed) > 1):
                    # p1 is the current point, p2 is the seed
                    p1 = points[i][0]
                    p2 = seed[0]
                    dist = euclideanDistance(p1, p2)
                    if (dist < minDist):
                        minDist = dist
                        minCluster = [seed[1]]
                    elif (dist == minDist):
                        # another node has the same distance so add to the pool of classes
                        minCluster.append(seed[1])

            # settles ties if they exist
            if (len(minCluster) == 1):
                newCluster = minCluster[0]
                oldCluster = points[i][1]

                if (oldCluster != newCluster):
                    points[i][1] = newCluster
                    isDone = False

            elif (len(minCluster) > 1):
                #settles a tiebreaker by using random
                newCluster = int(minCluster[randint(0, len(minCluster) - 1)])
                oldCluster = points[i][1]

                if (oldCluster != newCluster):
                    points[i][1] = newCluster
                    isDone = False

            else:
                print("ERROR: Failed to find min cluster")

        if (graph > 4):
            figure += 1
            graph = 1
        plt.figure(figure)
        plt.subplot(2, 2, graph)            
        graph += 1
        
            
        plt.axis([0, 20, 0, 20])

        seedPoints, clusters = zip(*seeds[1:])
        seedsX, seedsY = zip(*seedPoints)
        for i in range(0, len(points)):
            
            if (pointsList[i] not in seedPoints):
                curCluster = points[i][1]
                c = colors[curCluster]

                plt.plot(pointsX[i], pointsY[i], color=c, marker='o')
            
        for i in range(1, len(seeds)):
            #print(seeds[i][1])
            curCluster = seeds[i][1]
            c = colors[curCluster]        
        
            plt.plot(seedsX[i-1], seedsY[i-1], color=c, marker='*')
                    

        seeds = createNewSeeds(points, seeds)
        timeout -= 1
                
    plt.show()


# In[1106]:

# finds the euclidean distance using the formula sqrt((x1 - x2)^2 + (y1 - y2)^2)
def euclideanDistance(p1, p2):
    distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return distance


# In[1107]:

import numpy as np

def createNewSeeds(points, seeds):
    
    # list formatting: [[(sumPoint), n, (muPoint) = sumPoint / n], seed = [(), (), n], ...]
    newSeeds = [[]]
    newPoints = [[]]
            
    for seed in seeds:
        if (len(seed) > 0):
            # inits the seeds to be zero
            newSeeds.append([seed[0], 1, seed[0]])
    
    for i in range(0, len(points)):
        cluster = points[i][1]
        point = points[i]
        sumPointArray = np.array(newSeeds[cluster][0])
        n = newSeeds[cluster][1]
        
        sumPointArray += np.array(point[0])
        sumPoint = tuple(sumPointArray)
        n += 1
        muPoint = (sumPoint[0] / n, sumPoint[1] / n)
        
        #print("Before: " + str(newSeeds[cluster][0]) + " " + str(newSeeds[cluster][1]) + " " + str(newSeeds[cluster][2]))
        
        newSeeds[cluster][0] = sumPoint
        newSeeds[cluster][1] = n
        newSeeds[cluster][2] = muPoint
        
        #print("After: " + str(newSeeds[cluster][0]) + " " + str(newSeeds[cluster][1]) + " " + str(newSeeds[cluster][2]))
        
            
    for cluster in range(1, len(newSeeds)):
        seed = newSeeds[cluster][2]            
        newPoints.append([newSeeds[cluster][2], cluster])        
    
    return newPoints


# In[1108]:

def loadFile(fileName):
    file = open(fileName, 'r')
    pointsX = []
    pointsY = []
    points = []
    for line in file:
        pairs = line.split(" ")
        x = float(pairs[0])
        y = float(pairs[1].strip('\n'))
        xyPair = (x, y)
        
        if ([xyPair, 0] not in points):
            points.append([xyPair, 0])
        
    file.close()
    return points


# In[1109]:

main()


# In[ ]:




# In[ ]:



