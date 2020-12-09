class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Node():
    def __init__(self, x0, y0, w, h, points):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self):
        return self.points


class QTree():
    def __init__(self, k, n):
        self.threshold = k
        self.points = [Point(x[0], x[1]) for x in n]
        self.root = Node(0, 0, 699, 639, self.points)

    def add_point(self,x, y):
        self.points.append(Point(x, y))

    def get_points(self):
        return self.points

    def subdivide(self):
        print('hrere')
        self.recursive_subdivide(self.root, self.threshold)
        print('no here')

    def graph(self):
        c = self.find_children(self.root)
        areas = set()
        for el in c:
            areas.add(el.width * el.height)
        x_grid = list()
        y_grid = list()
        for n in c:
            x_grid.append(n.x0)
            y_grid.append(n.y0)
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        return x_grid, y_grid

    def recursive_subdivide(self, node, k):
        if len(node.points) <= k:
            return

        w_ = float(node.width / 2)
        h_ = float(node.height / 2)

        p = self.contains(node.x0, node.y0, w_, h_, node.points)
        x1 = Node(node.x0, node.y0, w_, h_, p)
        self.recursive_subdivide(x1, k)

        p = self.contains(node.x0, node.y0 + h_, w_, h_, node.points)
        x2 = Node(node.x0, node.y0 + h_, w_, h_, p)
        self.recursive_subdivide(x2, k)

        p = self.contains(node.x0 + w_, node.y0, w_, h_, node.points)
        x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
        self.recursive_subdivide(x3, k)

        p = self.contains(node.x0 + w_, node.y0 + w_, w_, h_, node.points)
        x4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, p)
        self.recursive_subdivide(x4, k)

        node.children = [x1, x2, x3, x4]

    def contains(self, x, y, w, h, points):
        pts = []
        for point in points:
            if point.x >= x and point.x <= x + w and point.y >= y and point.y <= y + h:
                pts.append(point)
        return pts

    def find_children(self,node):
        if not node.children:
            return [node]
        else:
            children = []
            for child in node.children:
                children += (self.find_children(child))
        return children