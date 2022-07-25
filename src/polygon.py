import matplotlib.pyplot as plt
import numpy as np
# vertex form is (x, y, z) or (x, y)


class Polygon(object):
    def __init__(self):
        self.vertices = None
        self.position = None
        self.node_parent = None
        self.node_level = 0 # 0: root
        self.direction = None # 0: front, 1: back

    def signed_area(self):
        """Return the signed area enclosed by a ring in linear time using the
        algorithm at: https://web.archive.org/web/20080209143651/http://cgafaq.info:80/wiki/Polygon_Area
        """
        xs, ys = [], []
        vertices = self.get_vertices()

        for x, y in vertices:
            xs.append(x)
            yy.append(y)

        xs.append(vertices[0][0])
        ys.append(vertices[0][1])
        xs.append(vertices[1][0])
        ys.append(vertices[1][1])

        return sum(xs[i] * (ys[i+1] - ys[i-1]) for i in range(1, len(vertices) + 1)) / 2.0

    def _set_vertices(self, vertices):
        self.vertices = vertices

    def _set_position(self, position):
        self.position = position

    def set_vertices(self, vertices):
        assert vertices is not None
        self._set_vertices(vertices)
        self.set_position()

        vertices_ = self.sort_vertices(vertices, self.get_position())
        self._set_vertices(vertices_)
        self.set_position()

    def set_position(self):
        vertices = self.get_vertices()
        assert vertices is not None

        position = list(np.mean(vertices, axis=0))
        self._set_position(position)

    def sort_vertices(self, vertices, position):
        x = np.array(vertices).astype(np.float32)[:, 0]
        y = np.array(vertices).astype(np.float32)[:, 1]
        angles = np.arctan2(x - position[0], y - position[1])

        sort_tups = sorted([(i, j, k) for i, j, k in zip(x, y, angles)], key = lambda t: t[2])

        x, y, angles = zip(*sort_tups)
        x = list(x)
        y = list(y)

        return [[a, b] for a, b in zip(x, y)]

    def add_vertices(self, vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)

        self.set_vertices(self.get_vertices())

    def del_vertices(self, vertex):
        if vertex in self.vertices:
            self.vertices.remove(vertex)

        self.set_vertices(self.get_vertices())

    def get_vertices(self):
        return self.vertices

    def get_position(self):
        return self.position

    def get_bound(self):
        vertices = self.get_vertices()
        vertices_ = np.array(vertices)

        x_min = np.min(vertices_[:, 0])
        x_max = np.max(vertices_[:, 0])
        y_min = np.min(vertices_[:, 1])
        y_max = np.max(vertices_[:, 1])

        return x_min, x_max, y_min, y_max


if __name__ == '__main__':
    poly = Polygon()
    #coord = [[1,1], [2,1], [2,2], [1,2], [0.5,1.5]]
    coord = [[1,1], [2,2], [2,1], [1,2]]

    poly.set_vertices(coord)

    poly.add_vertices([0.5, 1.5])
#     vertices = poly.vertices.copy()
#     vertices.append(vertices[0])
#     print(vertices)
#     xs, ys = zip(*vertices)
#     print(poly.get_position())

#     plt.figure()
#     plt.plot(xs, ys)
#     plt.show()

    poly.del_vertices([0.5, 1.5])
    vertices = poly.vertices.copy()
    vertices.append(vertices[0])

    print(vertices)
    xs, ys = zip(*vertices)
    print(poly.get_position())
