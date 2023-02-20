import matplotlib.pyplot as plt
import numpy as np
from gpytoolbox import ray_mesh_intersect, read_mesh, per_face_normals, per_vertex_normals  # for ray-mesh intersection queries

# abstraction for every scene object
class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return
      
class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params):
        """
        Initializes a sphere object with its radius, position and diffuse albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays: output the
        intersection distances (set to np.inf if none), and unit hit
        normals (set to [np.inf, np.inf, np.inf] if none.)
        """
        distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        distances[:] = np.inf
        normals = np.zeros(rays.Os.shape, dtype=np.float64)
        normals[:,:] = np.array([np.inf, np.inf, np.inf])

        rays_copy = Rays(rays.Os, rays.Ds)
        A = np.linalg.norm(rays_copy.Ds,axis=1)**2
        B = 2*np.sum(rays_copy.Ds*(rays_copy.Os - self.c), axis = 1)
        C = np.linalg.norm(rays_copy.Os - self.c, axis = 1)**2 - self.r**2
        delta = B**2 - 4*A*C

        zeros = np.where(np.isclose(delta,0))
        negatives = np.where(np.logical_and(np.logical_not(np.isclose(delta,0)),delta<1))
        positives = np.where(np.logical_and(np.logical_not(np.isclose(delta,0)),delta>1)) 
        
        #get the minimum positive t value for delta>0
        t1 = (-1*B[positives] - np.sqrt(delta[positives]))/(2*A[positives])
        t2 = (-1*B[positives] + np.sqrt(delta[positives]))/(2*A[positives])

        target = np.copy(t1)

        ind = np.where(np.logical_and(t1<0,t2<0)) #sphere is behind
        target[ind] = np.inf
        ind = np.where(np.logical_and(t1<0,t2>0))  #we are inside the sphere
        target[ind] = t2[ind]
        ind = np.where(np.logical_and(t1>0,t2>0))  #spehere is infront of us
        target[ind] = np.minimum(t1[ind],t2[ind])

        #set values of t bases on the three conditions for delta
        distances[negatives],distances[positives],distances[zeros] = np.inf, target, (-1/2)*B[zeros]

        intersection_points = rays(distances) 
        normals = intersection_points-self.c
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms == 0, 1, norms)

        normals = normals/norms[:,None] 
        normals = np.float64(np.where(np.isnan(normals),0,normals))
        normals[negatives] = np.array([np.inf, np.inf, np.inf])
        # distances = np.float64(distances)        
        return distances, normals
      
# triangle mesh objects for our scene
class Mesh(Geometry):
    def __init__(self, filename, brdf_params):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        # self.face_normals = per_face_normals(self.v, self.f, unit_norm = True)
        self.face_normals = per_face_normals(self.v, self.f, unit_norm = True)
        ### END SOLUTION
        super().__init__()

    def intersect(self, rays):
        hit_normals = np.array([np.inf, np.inf, np.inf])
        hit_distances, triangle_hit_ids, barys = ray_mesh_intersect(rays.Os, rays.Ds, self.v,
                                                                    self.f, use_embree=True)
        # temp_normals = self.face_normals[triangle_hit_ids]
        v_normals = per_vertex_normals(self.v, self.f)
        faces = self.f[triangle_hit_ids]   #(h*w, 3)
        shape = faces.shape
        faces = faces.flatten()            #(h*w*3, )
        barys = barys.flatten()[:,None]    #(h*w*3,1)
       
        vertex_normals = v_normals[faces]   #(h*w*3, 3)
        phong_normals = (vertex_normals*barys).reshape((shape[0],shape[1],3))
        phong_normals = np.sum(phong_normals, axis = 1)
        phong_normals = np.where((triangle_hit_ids == -1)[:, np.newaxis], hit_normals, phong_normals)

        hit_normals = phong_normals

        # temp_normals = np.where((triangle_hit_ids == -1)[:, np.newaxis],
        #                         hit_normals,
        #                         temp_normals)
        # hit_normals = temp_normals

        return hit_distances, hit_normals
