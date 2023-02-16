import matplotlib.pyplot as plt
import numpy as np

# Shade a scene given a bundle of eye rays; outputs a color image suitable for matplotlib visualization
def shade(scene, rays):
    shadow_ray_o_offset = 1e-6
    L = L.reshape((scene.h, scene.w, 3))
    return L

if __name__ == "__main__":
    enabled_tests = [True, False, False, False, False]

    ##########################################
    #Test Rays Sphere Intersection
    ##########################################
    if enabled_tests[0]:
        # Point tests for ray-sphere intersection
        sphere = Sphere(1, np.array([0, 0, 0]))
        rays = Rays(np.array([
            # Moving ray origin along y-axis with x, z axis fixed
            [0, 2, -2],  # should not intersect
            [0, 1, -2],  # should intersect once (tangent)
            [0, 0, -2],  # should intersect twice
            [0, -1, -2],  # should intersect once (bottom)
            [0, -2, -2],  # should not intersect
            # Move back along the z-axis
            [0, 0, -4],  # should have t 2 greater than that of origin [0, 0, -2]
        ]), np.array([[0, 0, 1]]))

        expected_ts = np.array([np.inf, 2, 1, 2, np.inf, 3], dtype=np.float64)
        hit_distances = sphere.intersect(rays)

        if np.allclose(hit_distances, expected_ts):
            print("Rays-Sphere Intersection point test passed")
        else:
            raise ValueError(f'Expected intersection distances {expected_ts}\n'
                             f'Actual intersection distances {hit_distances}')

    ##########################################
    #Test Eye Ray Generation
    ##########################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        scene = Scene(w=1024, h=768)  # TIP: if you haven't yet vectorized your code, try debugging at a lower resolution
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(10, np.array([0, 0, 50]))

        vectorized_eye_rays = scene.generate_eye_rays()
        hit_distances = sphere.intersect(vectorized_eye_rays)

        # Visualize hit distances
        plt.matshow(hit_distances.reshape((768, 1024)))
        plt.title("Distances")
        plt.colorbar()
        plt.show()

    ##########################################
    #Test Rays Scene Intersection
    ##########################################
    if enabled_tests[2]:
        # Set up scene
        scene = Scene(w=1024, h=768)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_spheres([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]), rho=np.array([0.999, 0.5, 0.5])),
            # Left Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]), rho=np.array([0.5, 0.999, 0.5])),
            # Ground
            Sphere(1650, np.array([23, -1700, 140]), rho=np.array([0.7, 0.7, 0.7])),
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        hit_distances, hit_normals, hit_ids = scene.intersect(vectorized_eye_rays)

        # Visualize distances, normals and IDs
        plt.matshow(hit_distances.reshape((768, 1024)))
        plt.title("Distances")
        plt.show()
        plt.matshow(np.abs(hit_normals.reshape((768, 1024, 3))))
        plt.title("Normals")
        plt.show()
        plt.matshow(hit_ids.reshape((768, 1024)))
        plt.title("IDs")
        plt.show()

    # Test shading
    if enabled_tests[3]:
        # Set up scene
        scene = Scene(w=1024, h=768)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_spheres([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]), rho=np.array([0.999, 0.5, 0.5])),
            # Right Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]), rho=np.array([0.5, 0.999, 0.5])),
            # Ground
            Sphere(1650, np.array([23, -1700, 140]), rho=np.array([0.7, 0.7, 0.7])),
        ])
        scene.add_lights([
            {
                "type": "directional",
                # Top-Left of the scene
                "direction": normalize(np.array([1, 1, 0])),
                "color": np.array([2, 0, 0, 1])  # Red
            },
            {
                "type": "directional",
                # Top-Right of the scene
                "direction": normalize(np.array([-1, 1, 0])),
                "color": np.array([0, 2, 0, 1])  # Green
            },
            {
                "type": "directional",
                # Top of the scene
                "direction": normalize(np.array([0, 1, 0])),
                "color": np.array([2, 2, 2, 1])  # White
            },
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        L = shade(scene, vectorized_eye_rays)

        plt.matshow(L)
        plt.title("Rendered Image")
        # plt.savefig("numpy-image.png")
        plt.show()