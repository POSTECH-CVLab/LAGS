from polygon import Polygon
from algorithm import split_to_quadrangle
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage import color
import skimage.transform as skit
import argparse
import time
from datetime import datetime
from shapely.geometry import Polygon as splg
from shapely import affinity

import utils

plt.style.use('dark_background')


def generate_from_bottom_left(polys):
    polys_position = [poly.get_position() for poly in polys]

    sorted_polys_position = sorted(polys_position, key=lambda x: (x[1], x[0]), reverse=False)

    indices = []
    for poly_position in sorted_polys_position:
        index = polys_position.index(poly_position)
        indices.append(index)

    return indices

def generate_random_sequence(polys, remain_idx):
    if len(remain_idx) == 0:
        return []
    else:
        remain_idx_copy = remain_idx.copy()

    remain_polys = []
    for i in remain_idx_copy:
        remain_polys.append(polys[i])

    splg_leaf_polys = [splg(x.get_vertices()) for x in remain_polys]

    pop_frags_idx = []
    trans_matrix = [1, 0, 0, 1, 0, 0.01]

    for i, (splg_leaf_poly, r_idx) in enumerate(zip(splg_leaf_polys, remain_idx_copy)):
        remain_splg_leaf_polys = splg_leaf_polys.copy()
        remain_splg_leaf_polys.pop(i)

        trans_splg_leaf_poly = affinity.affine_transform(splg_leaf_poly, trans_matrix)

        check_disjoint = True
        for remain_splg_leaf_poly in remain_splg_leaf_polys:
            if not trans_splg_leaf_poly.disjoint(remain_splg_leaf_poly):
                check_disjoint = False
                break

        if check_disjoint:
            pop_frags_idx.append(r_idx)

    print('length {}'.format(len(pop_frags_idx)))
    if len(pop_frags_idx) == 0:
        indices_ = generate_from_bottom_left(remain_polys)
        pop_frags_idx.append(remain_idx[indices_[0]])

    rnd_idx = list(np.random.choice(list(pop_frags_idx).copy(), 1))
    pop_frag_idx = remain_idx_copy.index(rnd_idx[0])

    remain_idx_copy.pop(pop_frag_idx)
    remain_polys.pop(pop_frag_idx)

    return rnd_idx + generate_random_sequence(polys, remain_idx_copy)

def get_coord(str_target_shape, target_edge):
    coord = []
    if str_target_shape == 'square':
        coord += [
            [0, 0],
            [0, target_edge],
            [target_edge, target_edge],
            [target_edge, 0],
        ]
    elif str_target_shape == 'pentagon':
        edge = target_edge / (2 * np.cos(18 / 180 * np.pi))
        coord += [
            [(target_edge / 2) - edge * np.sin(36 / 180 * np.pi), 0],
            [0, edge * (np.cos(36 / 180 * np.pi) + np.sin(18 / 180 * np.pi))],
            [target_edge / 2, edge * (np.cos(36 / 180 * np.pi) + 1)],
            [target_edge, edge * (np.cos(36 / 180 * np.pi) + np.sin(18 / 180 * np.pi))],
            [(target_edge / 2) + edge * np.sin(36 / 180 * np.pi), 0],
        ]
    elif str_target_shape == 'hexagon':
        edge = target_edge / 2
        coord += [
            [edge, 0],
            [0, edge * np.cos(60 / 180 * np.pi)],
            [0, edge * (1 + np.cos(60 / 180 * np.pi))],
            [edge, target_edge],
            [target_edge, edge * (1 + np.cos(60 / 180 * np.pi))],
            [target_edge, edge * np.cos(60 / 180 * np.pi)],
        ]
    elif str_target_shape == 'octagon':
        pass

    return coord

def set_config(target_edge):
    plt.xlim((0, target_edge))
    plt.ylim((0, target_edge))
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

def get_image_arr(fig):
    buf = fig.canvas.buffer_rgba()
    xx = np.array(buf)
    gray_image = color.rgb2gray(xx)
    gray_image = gray_image.astype(np.float32)

    return xx, gray_image

def save_image(data_path, str_file, gray_image, save_figures):
    np.save(os.path.join(data_path, str_file + '.npy'), gray_image)
    if save_figures:
        plt.savefig(os.path.join(data_path, str_file + '.png'))


_, args = utils.get_parser()

num_of_partition = args.num_of_partition # number of sequence data samples
num_of_data = args.num_of_data
num_of_frags = 2**num_of_partition
target_shape = args.target_shape
save_figures = args.save_figures
dpi = args.dpi
str_datasets = args.str_datasets
use_random_seq = args.use_random_seq
use_rotation = args.use_rotation
max_seq = args.max_seq
split_perpen = args.split_perpen
target_edge = args.target_edge


if __name__ == '__main__':
    utils.print_info(args)

    np.random.seed(42)

    start_time = time.time()
    now = datetime.now()
    str_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")
    print("start time {}".format(str_datetime))

    data_path = utils.get_str_directory(str_datasets, target_shape, num_of_data, num_of_frags, use_random_seq, split_perpen, use_rotation)
    print(data_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # target Polygon
    poly = Polygon()

    coord = get_coord(target_shape, target_edge)
    poly.set_vertices(coord)
    coord.append(coord[0])

    fx, fy = zip(*coord)
    fig, ax = plt.subplots(figsize=(target_edge, target_edge), dpi=dpi) # dpi = 128 x 128 resolution
    ax.fill(fx, fy, facecolor='w', edgecolor=None, linewidth=0.0)

    set_config(target_edge)
    fig.canvas.draw()
    xx, gray_image = get_image_arr(fig)
    save_image(data_path, 'target_shape', gray_image, save_figures)
    plt.close(fig)

    orig_target_shape = gray_image.copy()

    target_position = poly.get_position() # center of saving image

    all_data = []
    all_data_remain = []

    for idx in range(0, num_of_data):
        if idx % 100 == 0:
            print('{}st in progress'.format(idx + 1))

        # split the polygon using quadrangle bsp
        leaf_polys = [poly]
        for i in range(0, num_of_partition):
            temp_polys = []

            for temp_poly in leaf_polys:
                temp_polys += split_to_quadrangle(temp_poly, is_perpen=split_perpen)
            leaf_polys = temp_polys

        # plot the result of splitting and save an array and an image of plot
        fig, ax = plt.subplots(figsize=(target_edge, target_edge), dpi=dpi)

        for c_poly in leaf_polys:
            c_poly_vertices = c_poly.vertices.copy()
            c_poly_vertices.append(c_poly_vertices[0])
            fx, fy = zip(*c_poly_vertices)            
            ax.fill(fx, fy, facecolor='w', edgecolor='w', linewidth=1)

        set_config(target_edge)
        fig.canvas.draw()
        xx, gray_image = get_image_arr(fig)
        save_image(data_path, 'total_shape_{}'.format(idx + 1), gray_image, save_figures)
        plt.close(fig)        

        list_indices = []
        if not use_random_seq:
            indices = generate_from_bottom_left(leaf_polys)
            list_indices.append(indices)
        else:
            for _ in range(0, max_seq):
                indices = generate_random_sequence(leaf_polys, list(range(0, len(leaf_polys))))
                indices = indices[::-1]

                if indices not in list_indices:
                    list_indices.append(indices)

        for ind_indices, indices in enumerate(list_indices):
            sorted_leaf_polys = []

            for ind in indices:
                sorted_leaf_polys.append(leaf_polys[ind])

            pickle_data = []
            pickle_data_remain = []

            # plot the sequence of splitted polygons and save an array and an image of plot
            fig_cum, ax_cum = plt.subplots(figsize=(target_edge, target_edge), dpi=dpi)

            for i, leaf_poly in enumerate(sorted_leaf_polys):
                leaf_poly_position = np.array(leaf_poly.get_position())
                leaf_poly_vertices = leaf_poly.vertices.copy()
                leaf_poly_vertices.append(leaf_poly_vertices[0])

                fx, fy = zip(*leaf_poly_vertices)
                ax_cum.fill(fx, fy, facecolor='w', edgecolor='w', linewidth=1)

                set_config(target_edge)
                fig_cum.canvas.draw()
                xx, gray_image_cum = get_image_arr(fig_cum)
                save_image(data_path, 'fragment_{}_{}_{}_cum'.format(idx + 1, ind_indices + 1, i + 1), gray_image_cum, save_figures)

                gap_position = target_position - leaf_poly_position

                trans_coord = list(map(lambda x: x + gap_position, leaf_poly.get_vertices()))

                poly_centered = Polygon()
                poly_centered.set_vertices(trans_coord)

                poly_centered_vertices = poly_centered.get_vertices().copy()
                poly_centered_vertices.append(poly_centered_vertices[0])

                fx, fy = zip(*poly_centered_vertices)
                fig, ax = plt.subplots(figsize=(target_edge, target_edge), dpi=dpi)
                ax.fill(fx, fy, facecolor='w', edgecolor='w', linewidth=1)

                set_config(target_edge)
                fig.canvas.draw()
                xx, gray_image = get_image_arr(fig)

                if use_rotation:
                    cur_angle = np.random.randint(0, 12)
                    gray_image = skit.rotate(gray_image, 30 * cur_angle)
                else:
                    cur_angle = 0

                save_image(data_path, 'fragment_{}_{}_{}_{}'.format(idx + 1, ind_indices + 1, i + 1, cur_angle), gray_image, save_figures)

                pickle_data.append([poly_centered, leaf_poly, gray_image, gray_image_cum, indices[i], cur_angle])
                pickle_data_remain.append([poly_centered, leaf_poly, gray_image, orig_target_shape - gray_image_cum, indices[i], cur_angle])

                plt.close(fig)
            plt.close(fig_cum)

            ##
            fig_remain, ax_remain = plt.subplots(figsize=(target_edge, target_edge), dpi=dpi)
            ax_remain.fill([0, target_edge, target_edge, 0], [0, 0, target_edge, target_edge], facecolor='white', edgecolor='white', linewidth=1)

            for i, leaf_poly in enumerate(sorted_leaf_polys):
                leaf_poly_position = np.array(leaf_poly.get_position())
                leaf_poly_vertices = leaf_poly.vertices.copy()
                leaf_poly_vertices.append(leaf_poly_vertices[0])

                fx, fy = zip(*leaf_poly_vertices)

                ax_remain.fill(fx, fy, facecolor='black', edgecolor='black', linewidth=1)
                set_config(target_edge)
                xx, gray_image_cum = get_image_arr(fig_remain)
                fig_remain.canvas.draw()
                save_image(data_path, 'fragment_{}_{}_{}_remain'.format(idx + 1, ind_indices + 1, i + 1), gray_image_cum, save_figures)

            plt.close(fig_remain)
            ##

#        pickle.dump(pickle_data, open(os.path.join(data_path, 'total_polygons_{}.pt'.format(idx + 1)), "wb"))
        all_data.append(pickle_data)
        all_data_remain.append(pickle_data_remain)

    pickle.dump(all_data, open(os.path.join(data_path, 'all_input_polygon_{}.pt'.format(num_of_data)), "wb"))
    pickle.dump(all_data_remain, open(os.path.join(data_path, 'all_input_polygon_remain_{}.pt'.format(num_of_data)), "wb"))
