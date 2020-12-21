import cv2
import numpy as np
import sys
from math import sqrt
import random
try:
    import Image
except ImportError:
    from PIL import Image

# Get Video file from given command line argument 
cap = cv2.VideoCapture(sys.argv[1])

window_name = 'Window'

fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if file is appropriate or not 
if cap.isOpened() == False:
    print('WRONG CODEC USED OR ERROR FILE NOT FOUND!')

# Set as FULL screen 
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# K-means - to extract most dominant RGB values from given samples 
class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates

class Cluster:
    def __init__(self, center, points):
        self.center = center
        self.points = points

class KMeans:
    def __init__(self, n_clusters, min_diff=1):
        self.n_clusters = n_clusters
        self.min_diff = min_diff

    def calculate_center(self, points):
        n_dim = len(points[0].coordinates)
        vals = [0.0 for i in range(n_dim)]
        for p in points:
            for i in range(n_dim):
                vals[i] += p.coordinates[i]
        coords = [v / len(points) for v in vals]
        return Point(coords)

    def assign_points(self, clusters, points):
        plists = [[] for i in range(self.n_clusters)]
        for p in points:
            smallest_distance = float('inf')
            for i in range(self.n_clusters):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)
        return plists

    def fit(self, points):
        clusters = [Cluster(center=p, points=[p]) for p in
                    random.sample(points, self.n_clusters)]
        while True:
            plists = self.assign_points(clusters, points)
            diff = 0
            for i in range(self.n_clusters):
                if not plists[i]:
                    continue
                old = clusters[i]
                center = self.calculate_center(plists[i])
                new = Cluster(center, plists[i])
                clusters[i] = new
                diff = max(diff, euclidean(old.center, new.center))

            if diff < self.min_diff:
                break
        return clusters

def euclidean(p, q):
    n_dim = len(p.coordinates)
    return sqrt(sum([(p.coordinates[i] - q.coordinates[i]) ** 2
                for i in range(n_dim)]))


def get_points(image_path):
    img = Image.open(image_path)
    img.thumbnail((200, 400))
    img = img.convert('RGB')
    (w, h) = img.size

    points = []
    for (count, color) in img.getcolors(w * h):
        for _ in range(count):
            points.append(Point(color))

    return points

def rgb_to_hex(rgb):
    return '#%s' % ''.join('%02x' % p for p in rgb)

def get_colors(filename, n_colors=1):
    points = get_points(filename)
    clusters = KMeans(n_clusters=n_colors).fit(points)
    clusters.sort(key=lambda c: len(c.points), reverse=True)
    rgbs = [map(int, c.center.coordinates) for c in clusters]
    return list(map(rgb_to_hex, rgbs))

# colorsA - stores hex values of 5 most dominant colors from Team A sample
colorsA = get_colors(sys.argv[2], n_colors=5)
# colorsB - stores hex values of 5 most dominant colors from Team B sample
colorsB = get_colors(sys.argv[3], n_colors=5)


while cap.isOpened():

	# Read current frame 
    (ret, frame) = cap.read()

    if ret == True:
        # Edge Boundaries
        up_y = 0
        low_y = 540
        left_x = 0
        right_x = 720
        left = 0
        right = 720
        up = 0
        low = 540

        frame = cv2.resize(frame, (720, 540))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Range for pitch color (green)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 0xFF, 0xFF])
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        edges = cv2.Canny(mask1, 50, 150, apertureSize=3)

        # Detect the pitch using Hough Line Transform        
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        # Get Pitch Boundaries for the current frame 
        for (rho, theta) in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 100 * -b)
            y1 = int(y0 + 100 * a)
            x2 = int(x0 - 100 * -b)
            y2 = int(y0 - 100 * a)

            if x2 - x1 != 0 and abs((y2 - y1) / (x2 - x1)) <= 1:

                nx1 = 100
                nx2 = 620

                ny1 = int(y1 + (y2 - y1) / (x2 - x1) * (nx1 - x1))
                ny2 = int(y2 + (y2 - y1) / (x2 - x1) * (nx2 - x2))

                if ny1 < 270 or ny2 < 270:
                    tmp = min(ny1, ny2)
                    if up_y != 0:
                        up_y = min(up_y, tmp)
                    else:
                        up_y = tmp
                else:
                    tmp = max(ny1, ny2)
                    if low_y != 540:
                        low_y = max(low_y, tmp)
                    else:
                        low_y = tmp

            if x2 - x1 == 0 or abs((y2 - y1) / (x2 - x1)) > 1:

                ny1 = 100
                ny2 = 440

                nx1 = int(x1 + (x2 - x1) / (y2 - y1) * (ny1 - y1))
                nx2 = int(x2 + (x2 - x1) / (y2 - y1) * (ny2 - y2))

                if nx1 < 360 or nx2 < 360:
                    tmp = min(nx1, nx2)
                    if left_x != 0:
                        left_x = min(left_x, tmp)
                    else:
                        left_x = tmp
                else:
                    tmp = max(nx1, nx2)
                    if right_x != 720:
                        right_x = max(right_x, tmp)
                    else:
                        right_x = tmp

        rect1 = mask1[up:low, 0:left_x]

        if left_x != 0:
            if rect1.mean() / 0xFF < 0.5:
                left = left_x

        rect2 = mask1[up:low, right_x:720]

        if right_x != 720:
            if rect2.mean() / 0xFF < 0.5:
                right = right_x

        rect3 = mask1[0:up_y, 0:720]

        if up_y != 0:
            if rect3.mean() / 0xFF < 0.5:
                up = up_y

        rect4 = mask1[low_y:540, 0:720]

        if low_y != 540:
            if rect4.mean() / 0xFF < 0.5:
                low = low_y

        # Detect players using Edge Detection and Contours
        frame2 = frame[up:low, left:right, 0:3]
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        edges = cv2.Canny(mask1, 50, 150, apertureSize=3)

        (ret, thresh1) = cv2.threshold(mask1, 127, 0xFF,cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh1, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        median = cv2.medianBlur(mask1, 5)
        (_, contours, hierarchy) = cv2.findContours(median,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Color Range for white
        lower_white = np.array([0, 0, 192])
        upper_white = np.array([0, 0, 0xFF])
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, lower_white, upper_white)


        # Color For Team A
        h1 = colorsA[0].lstrip('#')
        A = np.uint8([[[int(h1[4:6], 16), int(h1[2:4], 16),int(h1[0:2], 16)]]])
        hsvA = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
        lower_A = np.array([max(0,hsvA[0][0][0] - 10), 80, 80])
        upper_A = np.array([min(255,hsvA[0][0][0] + 10), 0xFF, 0xFF])
        maskA = cv2.inRange(hsv, lower_A, upper_A)


        # Color For Team B
        h2 = colorsB[0].lstrip('#')
        B = np.uint8([[[int(h2[4:6], 16), int(h2[2:4], 16),int(h2[0:2], 16)]]])
        hsvB = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
        lower_B = np.array([max(0,hsvB[0][0][0] - 10), 80, 80])
        upper_B = np.array([min(255,hsvB[0][0][0] + 10), 0xFF, 0xFF])
        maskB = cv2.inRange(hsv, lower_B, upper_B)

        # Draw rectangles of different colors around obtained contours for both the teams
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if h < 1.2 * w or h > 2.8 * w:
                continue

            rect1 = mask1[y:y + h, x:x + w]
            if rect1.mean() / 0xFF > 0.7:
                continue

            rect2 = mask2[y:y + h, x:x + w]
            if rect2.mean() / 0xFF > 0.1:
                continue

            if cv2.contourArea(contour) < 100 \
                or cv2.contourArea(contour) > 400:
                continue

            rectA = maskA[y:y + h, x:x + w]
            if rectA.mean() / 0xFF > 0.01:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0xFF, 0, 0), 2)
                continue

            rectB = maskB[y:y + h, x:x + w]
            if rectB.mean() / 0xFF > 0.01:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 0xFF), 2)
                continue

            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0xFF, 0), 2)

        cv2.imshow(window_name, frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:

        break

cap.release()
cv2.destroyAllWindows()
