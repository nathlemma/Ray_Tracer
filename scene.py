import matplotlib.pyplot as plt
import numpy as np
import ray
import sphere

# A class to encapsulate everything about a scene: image resolution, scene objects, light properties
class Scene(object):
    REFRACTIVE_INDEX_OUT = 1.0
    REFRACTIVE_INDEX_IN = 1.5

    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters()
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_objects()
        self.spheres = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_spheres(self, spheres):
        """ Adds a list of objects to the scene. """
        self.spheres.extend(spheres)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """
        w = self.w
        h = self.h 
        fov = np.radians(self.fov)

        aspect_ratio = w/h

        ind = np.arange(w*h) #i%w -> x , i//w->y  
        # y,x = np.arange(h),np.arange(w)
        y,x = np.indices((h,w))
        y_ndc,x_ndc = (y + 0.5)/h, (x + 0.5)/w 
        y_screen, x_screen = 1 - (2 * y_ndc), (2 * x_ndc) - 1
        y_c, x_c = y_screen * np.tan(fov/2), x_screen * aspect_ratio * np.tan(fov/2)

        zc = self.at - self.eye
        xc = np.cross(self.up,zc)
        yc = np.cross(zc,xc)

        c_to_w_matrix = np.zeros((4,4))
        c_to_w_matrix[3,3] = 1
        c_to_w_matrix[:3,0] = xc
        c_to_w_matrix[:3,1] = yc
        c_to_w_matrix[:3,2] = zc
        c_to_w_matrix[:3,3] = self.eye
        
        ones = z = np.ones((h,w))
        cam_s = np.stack([x_c,y_c,ones,ones],axis = 2).reshape(h*w,4,1)
        world_s = np.zeros((h*w,4,1))
        for i in range(h*w):
          world_s[i] = np.matmul(c_to_w_matrix,cam_s[i].reshape(4,1))

        return ray.Rays(self.eye,world_s.reshape(h*w,4)[:,3])

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_ids = np.array([-1])
        hit_distances = np.array([np.inf])
        hit_normals = np.array([np.inf, np.inf, np.inf])

        return hit_distances, hit_normals, hit_ids