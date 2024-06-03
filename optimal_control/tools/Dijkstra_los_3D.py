import numpy as np
import pandas as pd
import math

import tools.LineOfSight_3D as los

def InObstacle(x, y, z, obstacles=None):
    if obstacles is None:
        return False

    for species in obstacles.keys():

        if obstacles[species] is None:
            continue

        if species == 'spheres':
            for obs in obstacles["spheres"].values():
                X, Y, Z = obs["xyz"][0], obs["xyz"][1], obs["xyz"][2]
                if ((X-x)**2 + (Y-y)**2 + (Z-z)*2) <= obs['r']**2:
                    return True
            
        if species == 'cylinders':
            for obs in obstacles["cylinders"].values():
                X, Y = obs["xy"][0], obs["xy"][1]
                Z0, Zh = obs["z"][0], obs["z"][1]
                if (z <= Z0) and (Zh <= z):
                    continue
                if ((X-x)**2 + (Y-y)**2) <= (obs['r'])**2:
                    return True
    
    return False

def Dijkstra_los_3D(xrange, yrange, zrange, startlocation, endlocation, obstacles=None, discrWidth=1, marge=1):
    """
    Shortest path algorithm based on Dijkstra, but made more efficient using the line of sight approach. 
    """
    print('-----------------------------------------------------------')
    print('Dijkstra algorithm with LineOfSight initialised with:')
    print(' ')
    print('xrange   : {}'.format(xrange))
    print('yrange   : {}'.format(yrange))
    print('zrange   : {}'.format(zrange))
    print(' ')
    print('Startlocation    : {}'.format(startlocation))
    print('Endlocation      : {}'.format(endlocation))
    print('Obstacles        : {}'.format(obstacles))
    print(' ')
    print('-----------------------------------------------------------')

    xcount = math.ceil((xrange[1] - xrange[0]) * (1/discrWidth)) + 1
    ycount = math.ceil((yrange[1] - yrange[0]) * (1/discrWidth)) + 1 
    zcount = math.ceil((zrange[1] - zrange[0]) * (1/discrWidth)) + 1

    if (xcount <= 0) or (ycount <= 0) or (zcount <= 0):
        raise Exception("Ranges not sorted and/or equal to zero: xcount = {}, ycount = {}, zcount ={}".format(xcount, ycount, zcount))


    # Building a DataFrame containing the information for all vertices in the grid with each vertex carrying the following information:
    #    - its own location ('X', 'Y' 'Z')
    #    - the currently known shortest total distance to the startvertex ('distance_to_start'). Initialised as infinite/very large.
    #    - the location of the first vertex that is passed when traveling towards the startpoint ('sourceX' and 'sourceY')
    #      when a shorter path ('distance_to_start') is found, this source vertex can also be updated. Initialised as [-1,-1]
    #    - whether or not the minimum distance and source vertex can still be updated ('free'). All vertices are initially free.

    columns = ['free', 'X', 'Y', 'Z', 'distance_to_start', 'sourceX', 'sourceY', 'sourceZ', 'distance_to_end', 'distance_total']
    vertices_df = pd.DataFrame(index=[], columns=columns)

    startbeelinedist = np.sqrt((startlocation[0] - endlocation[0]) ** 2 + (startlocation[1] - endlocation[1]) ** 2 + (startlocation[2] - endlocation[2]) ** 2)
    startvertex = pd.DataFrame({'free': [True], 'X': [startlocation[0]], 'Y': [startlocation[1]], 'Z': [startlocation[2]], 'distance_to_start': [0.0], 'sourceX': [-111],
                           'sourceY': [-111], 'sourceZ': [-111], 'distance_to_end': [startbeelinedist], 'distance_total': [9999 + startbeelinedist]})
    vertices_df = pd.concat([vertices_df, startvertex], ignore_index=True)

    for X in range(xcount):
        for Y in range(ycount):
            for Z in range(zcount):
                x = (X + xrange[0]) * discrWidth
                y = (Y + yrange[0]) * discrWidth
                z = (Z + zrange[0]) * discrWidth
                #print(X, x, ' | ', Y, y, ' | ', Z, z)
                if (x,y,z) == (startlocation[0], startlocation[1], startlocation[2]):
                    continue
                if (x,y,z) == (endlocation[0], endlocation[1], endlocation[2]):
                    continue

                # Points in obstacles are not added
                if not InObstacle(x,y,z, obstacles):
                    if (x,y,z) == (startlocation[0], startlocation[1], startlocation[2]):
                        print('Startlocation ADDED')
                    if (x,y,z) == (endlocation[0], endlocation[1], endlocation[2]):
                        print('Endlocation ADDED')

                    beelinedist = np.sqrt((x - endlocation[0]) ** 2 + (y - endlocation[1]) ** 2 + (z - endlocation[2]) ** 2)
                    vertex = pd.DataFrame(
                        {'free': [True], 'X': [x], 'Y': [y], 'Z': [z], 'distance_to_start': [9999], 'sourceX': [-111],
                        'sourceY': [-111], 'sourceZ': [-111], 'distance_to_end': [beelinedist], 'distance_total': [9999 + beelinedist]})
                    #vertices_df = vertices_df.append(vertex, ignore_index=True)
                    vertices_df = pd.concat([vertices_df, vertex], ignore_index=True)

    endvertex = pd.DataFrame({'free': [True], 'X': [endlocation[0]], 'Y': [endlocation[1]], 'Z': [endlocation[2]], 'distance_to_start': [9999], 'sourceX': [-111],
                           'sourceY': [-111], 'sourceZ': [-111], 'distance_to_end': [0.0], 'distance_total': [9999]})
    vertices_df = pd.concat([vertices_df, endvertex], ignore_index=True)

    # Find the label of the start- and endlocation, which will be usefull later on.
    startindex = vertices_df[(vertices_df['X'] == startlocation[0]) & (vertices_df['Y'] == startlocation[1]) & (vertices_df['Z'] == startlocation[2])].index[0]
    endindex = vertices_df[(vertices_df['X'] == endlocation[0]) & (vertices_df['Y'] == endlocation[1]) & (vertices_df['Z'] == endlocation[2])].index[0]

    # The startlocation is assigned a distance of 0 and is considered as its own source vertex.
    #vertices_df.loc[(vertices_df['X'] == startlocation[0]) & (vertices_df['Y'] == startlocation[1]) & (vertices_df['Z'] == startlocation[2]), 'free'] = False
    vertices_df.loc[(vertices_df['X'] == startlocation[0]) & (vertices_df['Y'] == startlocation[1]) & (vertices_df['Z'] == startlocation[2]), 'distance_to_start'] = 0.0
    vertices_df.loc[(vertices_df['X'] == startlocation[0]) & (vertices_df['Y'] == startlocation[1]) & (vertices_df['Z'] == startlocation[2]), 'distance_total'] = 0.0
    vertices_df.loc[(vertices_df['X'] == startlocation[0]) & (vertices_df['Y'] == startlocation[1]) & (vertices_df['Z'] == startlocation[2]), ['sourceX', 'sourceY', 'sourceZ']] = startlocation


    # Using Dijkstra's algorithm to find the shortest path from the startlocation to the endlocation by
    # exploring the vertices around the startlocation. The vertex for which the currently known best distance to the starting
    # position is the lowest is fixed ('free' = False) and the distances for its neighboring vertices are updated.
    # If the distance through the considered vertex to its neighboring vertex is shorter than the currently known best distance
    # to that neighboring vertex, the vertex becomes its neighbor's source vertex
    # The loop is repeated until there are no free vertices left or when the vertex on the endlocation is fixed.

    while (vertices_df[vertices_df['free'] == True].shape[0] != 0):
        # Q is the set of vertices that are still free. Get the index of the free vertex with the lowest 'distance_to_start' value, set this vertex to not-free and store location.
        Q = vertices_df[vertices_df['free'] == True]
        currentindex = Q.astype(float).idxmin()['distance_total']

        vertices_df.loc[currentindex, 'free'] = False
        currentvertexX = vertices_df.loc[currentindex, 'X']
        currentvertexY = vertices_df.loc[currentindex, 'Y']
        currentvertexZ = vertices_df.loc[currentindex, 'Z']
        currentvertexdist = vertices_df.loc[currentindex, 'distance_to_start']

        # If the endlocation is set to fixed, the shortest path is found and we can leave the loop.
        if currentindex == endindex:
            print('Stopcondition satisfied')
            break
    
        # Update the set of free vertices Q
        Q = vertices_df[vertices_df['free'] == True]

        # Check for all free vertices if they are close enough (<= discrWidth*sqrt(2)) and if so, update 'distance_to_start' and source coordinates.
        for vertexindex in Q.index:
            neighborvertexX = Q.loc[vertexindex, 'X']
            neighborvertexY = Q.loc[vertexindex, 'Y']
            neighborvertexZ = Q.loc[vertexindex, 'Z']

            neighborvertexdist = Q.loc[vertexindex, 'distance_to_start']
            sourcevertexdist = np.sqrt((neighborvertexX - currentvertexX) ** 2 + (neighborvertexY - currentvertexY) ** 2 + (neighborvertexZ - currentvertexZ) ** 2)

            if los.lineofsight(vertices_df, currentindex, vertexindex, obstacles, marge):
                if currentvertexdist + sourcevertexdist < neighborvertexdist:
                    # Path through current vertex is shorter, so update the neighboring vertex
                    vertices_df.loc[vertexindex, 'sourceX'] = currentvertexX
                    vertices_df.loc[vertexindex, 'sourceY'] = currentvertexY
                    vertices_df.loc[vertexindex, 'sourceZ'] = currentvertexZ

                    vertices_df.loc[vertexindex, 'distance_to_start'] = currentvertexdist + sourcevertexdist
                    vertices_df.loc[vertexindex, 'distance_total'] = vertices_df.loc[vertexindex, 'distance_to_start'] + vertices_df.loc[vertexindex, 'distance_to_end']

    # If the endlocation cannot be reached and all possible points are consumed.
    if (vertices_df.loc[endindex, 'distance_to_start'] == 9999):
        raise Exception("All points fixed except endpoint. Endpoint cannot be reached with chosen obstacles and/or grid discretisation width")

    minimumDistance = vertices_df.loc[endindex, 'distance_to_start']
    fixedVertices = vertices_df[vertices_df['free'] == False].shape[0]

    #print('The minimum distance is: ' + str(vertices_df.loc[endindex, 'distance_to_start']))
    #print('The number of fixed vertices: ' + str(vertices_df[vertices_df['free'] == False].shape[0]))

    # P is the set of vertices that are fixed.
    P = vertices_df[vertices_df['free'] == False]

    endpointindex = P[(P['X'] == endlocation[0]) & (P['Y'] == endlocation[1]) & (P['Z'] == endlocation[2])].index[0]
    startpointindex = P[(P['X'] == startlocation[0]) & (P['Y'] == startlocation[1]) & (P['Z'] == startlocation[2])].index[0]

    endpoint = (P.loc[endpointindex, 'X'], P.loc[endpointindex, 'Y'], P.loc[endpointindex, 'Z'])
    startpoint = (P.loc[startpointindex, 'X'], P.loc[startpointindex, 'Y'], P.loc[startpointindex, 'Z'])

    route = []
    timefracroute = []

    nextpoint = endpoint
    nexttimefrac = 1

    while nextpoint != startpoint:
        route.append(nextpoint)

        nextpointindex = P[(P['X'] == nextpoint[0]) & (P['Y'] == nextpoint[1]) & (P['Z'] == nextpoint[2])].index[0]

        nextpoint = (P.loc[nextpointindex, 'sourceX'], P.loc[nextpointindex, 'sourceY'], P.loc[nextpointindex, 'sourceZ'])
        nexttimefrac = P.loc[nextpointindex, 'distance_to_start'] / P.loc[nextpointindex, 'distance_total']

        timefracroute.append(nexttimefrac)

    route.append(startpoint)
    timefracroute.append(0.0)


    return route, timefracroute,  minimumDistance, fixedVertices


"""
spheres = {'sphere_1': {'r': 0.5, 'xyz': [5, 3, 1]}}
cylinders = {'cil_1': {'r': 1, 'z': [0,4], 'xy': [8,5]}}

obstacles = {'spheres': spheres, 'cylinders': cylinders}


startlocation = [0, 0, 0]
endlocation = [10, 6, 4]

xrange = (0,10)
yrange = (0,6)
zrange = (0,4)

discrWidth = 1

print(InObstacle(0,0,0,obstacles))
print(InObstacle(10,6,3,obstacles))

route, timefracroute, minimumDistance, fixedVertices = Dijkstra_los_3D(xrange, yrange, zrange, startlocation, endlocation, obstacles, discrWidth)

print(route)
"""
