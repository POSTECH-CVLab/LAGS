from polygon import Polygon
import numpy as np


def get_edge(choice, parents_vertices, ind):
    if choice[ind] == len(parents_vertices) - 1:
        point = [parents_vertices[choice[ind]], parents_vertices[0]]
    else:
        point = [parents_vertices[choice[ind]], parents_vertices[choice[ind] + 1]]

    return point

def get_vertices(edge, rand_n):
    if (edge[1][0] - edge[0][0]) == 0:
        x = edge[1][0]
        y = (edge[1][1] - edge[0][1]) * rand_n + edge[0][1]
    else:
        slope_coef = (edge[1][1] - edge[0][1]) / (edge[1][0] - edge[0][0])
        x = (edge[1][0] - edge[0][0]) * rand_n + edge[0][0]
        y = slope_coef * (x - edge[0][0]) + edge[0][1]

    return [x, y]

def split(polygon):
    parents_vertices = polygon.get_vertices()
    choice = np.random.choice(
        range(0, len(parents_vertices)),
        2,
        replace=False
    )

    edge_1 = get_edge(choice, parents_vertices, 0)
    edge_2 = get_edge(choice, parents_vertices, 1)

    rand_n = np.random.rand(1)
    vertices_1 = get_vertices(edge_1, rand_n)

    rand_n = np.random.rand(1)
    vertices_2 = get_vertices(edge_2, rand_n)

    a = vertices_2[1] - vertices_1[1]
    b = -(vertices_2[0] - vertices_1[0])
    d = -a * vertices_1[0] - b * vertices_1[1]

    front_vertices = []
    back_vertices = []

    for vertex in parents_vertices:
        side = a * vertex[0] + b * vertex[1] + d

        if side >= 0:
            front_vertices.append(vertex)
        else:
            back_vertices.append(vertex)

    front_polygon = Polygon()
    back_polygon = Polygon()

    front_vertices.append(vertices_1)
    front_vertices.append(vertices_2)
    back_vertices.append(vertices_1)
    back_vertices.append(vertices_2)

    front_polygon.set_vertices(front_vertices)
    back_polygon.set_vertices(back_vertices)

    front_polygon.node_parent = polygon
    front_polygon.node_level = polygon.node_level + 1
    front_polygon.direction = 0

    back_polygon.node_parent = polygon
    back_polygon.node_level = polygon.node_level + 1
    back_polygon.direction = 1

    return front_polygon, back_polygon

def split_to_quadrangle(polygon: Polygon, is_perpen=False): # target_geo_shape : 4, 5, 6, 8 
    parents_vertices = polygon.get_vertices()
    choice = np.random.choice(range(len(parents_vertices)),
        1,
        replace=False
    )

    two_parents_vertices = parents_vertices + parents_vertices
    edge_num = len(parents_vertices)

    if edge_num == 4:
        gap = 2
    elif edge_num == 5:
        gap = np.random.choice([2, 3], 1)[0]
    elif edge_num == 6:
        gap = np.random.choice([2, 3, 4], 1)[0]
    elif edge_num == 8:
        gap = np.random.choice([3, 4, 5], 1)[0]
    else:
        raise ValueError('edge_num {}'.format(edge_num))

    edge_1 = [
        two_parents_vertices[choice[0]],
        two_parents_vertices[choice[0] + 1]
    ]
    edge_2 = [
        two_parents_vertices[choice[0] + gap],
        two_parents_vertices[choice[0] + gap + 1]
    ]

    rand_n = np.random.random() / 2. + 0.25
    vertices_1 = get_vertices(edge_1, rand_n)

    if is_perpen:
        rand_n = 1.0 - rand_n
    else:
        rand_n = np.random.random() / 2. + 0.25
    
    vertices_2 = get_vertices(edge_2, rand_n)

    a = vertices_2[1] - vertices_1[1]
    b = -(vertices_2[0] - vertices_1[0])
    d = -a * vertices_1[0] - b * vertices_1[1]

    front_vertices = []
    back_vertices = []

    for vertex in parents_vertices:
        side = a * vertex[0] + b * vertex[1] + d

        if side >= 0:
            front_vertices.append(vertex)
        else:
            back_vertices.append(vertex)

    front_polygon = Polygon()
    back_polygon = Polygon()

    front_vertices.append(vertices_1)
    front_vertices.append(vertices_2)
    back_vertices.append(vertices_1)
    back_vertices.append(vertices_2)

#    print('len(front_vertices) =', len(front_vertices))
#    print('front_vertices =', front_vertices)
#    print('len(back_vertices) =', len(back_vertices))
#    print('back_vertices =', back_vertices)

    front_polygon.set_vertices(front_vertices)
    back_polygon.set_vertices(back_vertices)

    front_polygon.node_parent = polygon
    front_polygon.node_level = polygon.node_level + 1
    front_polygon.direction = 0

    back_polygon.node_parent = polygon
    back_polygon.node_level = polygon.node_level + 1
    back_polygon.direction = 1

    return [front_polygon, back_polygon]
