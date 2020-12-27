from math import sqrt, floor
import numpy as np
import sys

"""
Create cluster centroids using the k-means++ algorithm.
Parameters
----------
ds : numpy array
    The dataset to be used for centroid initialization.
k : int
    The desired number of clusters for which centroids are required.
Returns
-------
centroids : numpy array
    Collection of k centroids as a numpy array.
Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
"""
def plus_plus(ds, k, random_state=42):
    np.random.seed(random_state)
    centroids = [ds[0]]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        print("dist_sq ", dist_sq,", probs ",probs,", cumuluative_probs ",cumulative_probs)
        for j, p in enumerate(cumulative_probs):
            print("j p ->",j,p)
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)

def distance(p1, p2): 
    return np.sum((p1 - p2)**2)

def initialize(data, k): 
    ''' 
    initialized the centroids for K-means++ 
    inputs: 
        data - numpy array of data points having shape (200, 2) 
        k - number of clusters  
    '''
    ## initialize the centroids list and add 
    ## a randomly selected data point to the list 
    centroids = [] 
    centroids.append(data[np.random.randint( 
            data.shape[0]), :]) 
    #plot(data, np.array(centroids)) 
   
    ## compute remaining k - 1 centroids 
    for c_id in range(k - 1): 
          
        ## initialize a list to store distances of data 
        ## points from nearest centroid 
        dist = [] 
        for i in range(data.shape[0]): 
            point = data[i, :] 
            d = sys.maxsize 
              
            ## compute distance of 'point' from each of the previously 
            ## selected centroid and store the minimum distance 
            for j in range(len(centroids)): 
                temp_dist = distance(point, centroids[j]) 
                d = min(d, temp_dist) 
            dist.append(d) 
              
        ## select data point with maximum distance as our next centroid 
        dist = np.array(dist) 
        next_centroid = data[np.argmax(dist), :] 
        centroids.append(next_centroid) 
        dist = [] 
        #plot(data, np.array(centroids)) 
    return centroids 

if __name__ == "__main__":
    test = plus_plus(np.array([0,1,2,3,4]),3)
    print(test)

    test = initialize(np.array([0,1,2,3,4]),3)
    print(test)
    