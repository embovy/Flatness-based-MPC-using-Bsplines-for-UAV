import numpy as np

def lineofsight(vertices_df, index1, index2, obstacles=None, marge=1):
    x1 = vertices_df.loc[index1, 'X']
    y1 = vertices_df.loc[index1, 'Y']
    z1 = vertices_df.loc[index1, 'Z']
    x2 = vertices_df.loc[index2, 'X']
    y2 = vertices_df.loc[index2, 'Y']
    z2 = vertices_df.loc[index2, 'Z']

    if obstacles is None:
        return True

    for species in obstacles.keys():

        if obstacles[species] is None:
            continue

        if species == 'spheres':
            for obs in obstacles["spheres"].values():
                r = obs["r"]
                X, Y, Z = obs["xyz"][0], obs["xyz"][1], obs["xyz"][2]
                if collisionSphere(x1, y1, z1, x2, y2, z2, X, Y, Z, r, marge=marge):
                    return False    
        
        if species == 'cylinders':
            for obs in obstacles["cylinders"].values():
                R = obs["r"]
                X, Y = obs["xy"][0], obs["xy"][1]
                Z0, Zh = obs["z"][0], obs["z"][1]
                if collisionCylinder(x1, y1, z1, x2, y2, z2, X, Y, Z0, Zh, R, marge=marge):
                    return False
                
    return True

def collisionSphere(x1, y1, z1, x2, y2, z2, X, Y, Z, r, marge):
    k = ((x2-x1)*(X-x1) + (y2-y1)*(Y-y1) + (z2-z1)*(Z-z1)) / ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + 1e-6)

    x, y, z = x1 + k*(x2-x1), y1 + k*(y2-y1), z1 + k*(z2-z1)

    if (X-x)**2 + (Y-y)**2 + (Z-z)**2 < (r + marge)**2:
        return True
    
    return False

def collisionCylinder(x1, y1, z1, x2, y2, z2, X, Y, Z0, Zh, R, marge, N=1000):
    K = np.linspace(0, 1, N)

    for k in K:
        x, y, z = x1 + k*(x2-x1), y1 + k*(y2-y1), z1 + k*(z2-z1)
        if (z <= Z0) and (Zh <= z):
            continue
        if ((X-x)**2 + (Y-y)**2) <= (R+marge)**2:
            return True
        
    return False


