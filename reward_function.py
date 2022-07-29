import numpy as np
from numpy import exp,sqrt,abs
from math import atan,degrees,ceil
def reward_function(params):
    x = params['x']
    y = params['y']
    waypoints=params['waypoints']
    next_index=params['closest_waypoints'][1]
    progress=params['progress']
    steering_angle=params['steering_angle']
    heading=params['heading']
    steps=params['steps']
    track_length=params['track_length']
    stepsK = round(3.0 * track_length/18,1)
    track_width = params['track_width']
    speed = params['speed']
    reward_base = 1e-2
    curPoint = (x,y)
    def getAngleBy2Point(Point1,Point2,tholdAngle=1e-5):
        deltaX = Point2[0] - Point1[0]
        deltaY = Point2[1] - Point1[1]
        direct = 0
        if np.abs(deltaX) < tholdAngle:
            track_direction = np.sign(deltaX) * np.pi/2
        if deltaX > tholdAngle:
            if deltaY > tholdAngle:
                direct = 1
                track_direction = atan(deltaY/deltaX)
            else:
                direct = 4
                track_direction = atan(deltaY/deltaX) + 2 * np.pi
        else:
            if deltaY > tholdAngle:
                direct = 2
                track_direction = atan(deltaY/deltaX) + np.pi
            else:
                direct = 3
                track_direction = atan(deltaY/deltaX) + np.pi
        track_direction = degrees(track_direction)
        return direct,track_direction
    def menger_curvature(pt1, pt2, pt3, atol=1e-3):
        vec21 = np.array([pt1[0]-pt2[0], pt1[1]-pt2[1]])
        vec23 = np.array([pt3[0]-pt2[0], pt3[1]-pt2[1]])
        norm21 = np.linalg.norm(vec21)
        norm23 = np.linalg.norm(vec23)
        if np.abs(np.dot(vec21, vec23)) < atol and norm21*norm23 < atol:
            return 0
        theta = np.arccos(np.dot(vec21, vec23)/(norm21*norm23))
        if np.abs(theta-np.pi) < atol:
            theta = 0.0
        dist13 = np.linalg.norm(vec21-vec23)
        if np.abs(dist13) < atol:
            return 0
        return 2*np.sin(theta) / dist13
    def getDistanceBy2Point(point1,point2):
        x1,y1 = point1
        x2,y2 = point2
        vec = (x2-x1,y2-y1)
        distance = sqrt(vec[0]**2+vec[1]**2)
        return distance
    def getDistanceByLineAndPoint(line,point,thold=1e-3):
        p1,p2 = line
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = point
        if np.abs(x1-x2) < thold and np.abs(y1-y2) < thold:
            return getDistanceBy2Point(((x1+x2)/2,(y1+y2)/2),point)
        elif np.abs(x1-x2) < thold:
            x_ = (x1+x2)/2
            return np.abs(x_-x3)
        elif np.abs(y1-y2) < thold:
            y_ = (y1+y2)/2
            return np.abs(y_-y3)
        else:
            return np.abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1 - x1*y2) / sqrt((y2-y1)**2 + (x2-x1)**2)
    def stepRewardK(progress,steps,stepsK,k=0):
        k = float(progress * stepsK/steps)
        k = 2.5 if k > 2.5 else k
        ret = exp(k**2)
        return ret
    def positionRewardK(curPoint,next_index,speed,track_width,race_line,mode,steering_angle,heading,tholdAngle=1e-5,angleThold = 10):
        delta = ceil(speed)
        line = (race_line[next_index%mode],race_line[(next_index+delta)%mode])
        next_point0 = race_line[(next_index)%mode]
        next_point1 = race_line[(next_index+delta)%mode]
        next_point2 = race_line[(next_index+delta*2)%mode]
        direct0,track_direction0 = getAngleBy2Point(next_point0, next_point2)
        direct1,track_direction1 = getAngleBy2Point(next_point1, next_point2)
        track_direction = (track_direction0+track_direction1) / 2.0
        heading = heading if heading > 0 else heading + 360
        direction_diff = abs(track_direction - heading)
        kAngle = angleThold/(direction_diff+10)
        distance = getDistanceByLineAndPoint(line,curPoint,thold=1e-3)
        k  = 0.8 - distance * 2.0/track_width
        k = -0.2 if k < -0.2 else k
        c0 = menger_curvature(curPoint, next_point0, next_point2, atol=1e-3)
        c1 = menger_curvature(curPoint, next_point1, next_point2, atol=1e-3)
        c = (c0 + c1) / 2.0
        vmax = sqrt(2.5/(c+0.05))
        vmax = 4 if vmax > 4 else vmax
        speedK = 1.4
        if speed < vmax:
            ret = exp(speed**speedK)
        else:
            ret = exp(speed**speedK)- exp(vmax**speedK)
        ret *= exp(k+kAngle)
        return distance,ret
    mode = len(waypoints)-1
    k1 = stepRewardK(progress,steps,stepsK)
    distance,k2 = positionRewardK(curPoint,next_index,speed,track_width,waypoints,mode,steering_angle,heading)
    reward = float(reward_base * (k1 + k2))
    return reward
