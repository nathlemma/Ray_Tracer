import matplotlib.pyplot as plt
import numpy as np

# A sphere object encapsulating the geometry and its diffuse material properties
class Sphere(object):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, rho=np.zeros((3,), dtype=np.float64)):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.rho = np.copy(rho)

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays, and compute the
        distance between the hit point on the sphere surface and the
        ray origins. If a ray did not intersect the sphere, set the
        distance to np.inf.
        """
        rays_copy = Rays(rays.Os, rays.Ds) #copy rays object to prevent modifying input object
        ddim,odim = rays.Ds.shape[0],rays.Os.shape[0]
        #fix dimensions
        if not(ddim==odim): 
          if(ddim==1):
            rays_copy.Ds = np.vstack([rays.Ds[0]]*odim)
          if(odim==1): 
            rays_copy.Os = np.vstack([rays.Os[0]]*ddim)

        A = np.linalg.norm(rays_copy.Ds,axis=1)**2
        B = 2*np.sum(rays_copy.Ds*(rays_copy.Os - self.c), axis = 1)
        C = np.linalg.norm(rays_copy.Os - self.c, axis = 1)**2 - self.r**2
        delta = B**2 - 4*A*C

        zeros = np.where(np.isclose(delta,0))
        negatives = np.where(np.logical_and(np.logical_not(np.isclose(delta,0)),delta<1))
        positives = np.where(np.logical_and(np.logical_not(np.isclose(delta,0)),delta>1)) 
        
        #to be returned
        t = np.array([0]*delta.shape[0], dtype = np.float64)  
        
        #get the minimum positive t value for delta>0
        t1 = (-1*B[positives] - np.sqrt(delta[positives]))/(2*A[positives])
        t2 = (-1*B[positives] + np.sqrt(delta[positives]))/(2*A[positives])

        target = np.copy(t1)

        ind = np.where(np.logical_and(t1<0,t2<0))
        target[ind] = np.inf
        ind = np.where(np.logical_and(t1<0,t2>0))
        target[ind] = t2[ind]
        ind = np.where(np.logical_and(t1>0,t2>0))
        target[ind] = np.minimum(t1[ind],t2[ind])

        #set values of t bases on the three conditions for delta
        t[negatives],t[positives],t[zeros] = np.inf, target, (-1/2)*B[zeros] 
        return t
        
