import numpy as np
import os
import pickle

import utils


_, args = utils.get_parser()

num_of_partition = args.num_of_partition
num_of_data = args.num_of_data
num_of_frags = 2**num_of_partition
target_shape = args.target_shape
save_figures = args.save_figures
dpi = args.dpi
str_datasets = args.str_datasets
use_random_seq = args.use_random_seq
max_seq = args.max_seq
split_perpen = args.split_perpen
use_rotation = args.use_rotation


if __name__ == '__main__':
    utils.print_info(args)

    np.random.seed(42)

    data_path = utils.get_str_directory(str_datasets, target_shape, num_of_data, num_of_frags, use_random_seq, split_perpen, use_rotation)
    print(data_path)

    train_data_path = os.path.join(data_path, 'train_remain')
    test_data_path = os.path.join(data_path, 'test_remain')

    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    total_num = int(num_of_data * num_of_frags)
    train_num = int(0.8 * total_num)

    print('train_num =', train_num)
    print('total_num =', total_num)

    target_shape = np.load(open(os.path.join(data_path, 'target_shape.npy'), 'rb'), allow_pickle=True)

    all_data_remain = pickle.load(open(os.path.join(data_path, 'all_input_polygon_remain_{}.pt'.format(num_of_data)), 'rb'))

    print('len(all_data_remain) =', len(all_data_remain))

#    [poly_centered, leaf_poly, gray_image, orig_target_shape - gray_image_cum, index]

    total_cur_plus_frags = []
    total_poly_centered = []
    total_next_frag = []
    total_next_frag_pos = []
    total_next_frag_ind = []
    total_cur_shape = []
    total_angle = []

    for i, polygons in enumerate(all_data_remain):
        if i % 100 == 0:
            print('{}st processing'.format(i + 1))

        remaining_shape = target_shape.copy()

        fragments = [list_data[2] for list_data in polygons]
        list_of_next_frags_pos = []

        for j, list_data in enumerate(polygons):
            if len(list_data) == 5:
                poly_centered, poly_orig, image_poly_orig, current_shape, index = list_data
                angle = 0
            elif len(list_data) == 6:
                poly_centered, poly_orig, image_poly_orig, current_shape, index, angle = list_data
            else:
                raise ValueError()

            total_poly_centered.append(poly_centered)
            cur_plus_frags = [remaining_shape] + fragments
            cur_plus_frags = np.array(cur_plus_frags).astype(np.float32)

            total_cur_plus_frags.append(cur_plus_frags)
            total_cur_shape.append(remaining_shape)

            remaining_shape = current_shape

            total_next_frag.append(image_poly_orig)
            total_next_frag_ind.append(index)
            total_angle.append(angle)

            next_frag_pos = poly_orig.get_position()
            t_x = int(current_shape.shape[1] - next_frag_pos[1] * current_shape.shape[1])
            t_y = int(next_frag_pos[0] * current_shape.shape[0])

            next_frag_pos_image = np.zeros_like(current_shape)
            next_frag_pos_image[t_x][t_y] = 1.0
            total_next_frag_pos.append(next_frag_pos_image)

    total_cur_plus_frags = np.array(total_cur_plus_frags)
    print('total_cur_plus_frags.shape', total_cur_plus_frags.shape)

    print(total_angle[:10])

    train_total_cur_plus_frags = total_cur_plus_frags[:train_num]
    test_total_cur_plus_frags = total_cur_plus_frags[train_num:]

    np.save(os.path.join(train_data_path, 'train_total_cur_plus_frags.npy'), train_total_cur_plus_frags)
    np.save(os.path.join(test_data_path, 'test_total_cur_plus_frags.npy'), test_total_cur_plus_frags)

    del train_total_cur_plus_frags
    del test_total_cur_plus_frags
    del total_cur_plus_frags

    total_next_frag = np.array(total_next_frag).astype(np.float32)
    print('total_next_frag.shape', total_next_frag.shape)

    total_next_frag_ind = np.array(total_next_frag_ind).astype(np.float32)
    print('total_next_frag_ind.shape', total_next_frag_ind.shape)

    total_cur_shape = np.array(total_cur_shape).astype(np.float32)
    print('total_cur_shape.shape', total_cur_shape.shape)

    total_angle = np.array(total_angle).astype(np.int32)
    print('total_angle.shape', total_angle.shape)

    print('len(total_poly_centered)', len(total_poly_centered))

    total_next_frag_pos = np.array(total_next_frag_pos).astype(np.float32)
    print('total_next_frag_pos.shape', total_next_frag_pos.shape)

    label_data = np.tile(np.arange(num_of_frags), (num_of_data, )).astype(np.float32)
    print('label_data.shape =', label_data.shape)

    train_total_next_frag = total_next_frag[:train_num]
    test_total_next_frag = total_next_frag[train_num:]

    train_total_next_frag_ind = total_next_frag_ind[:train_num]
    test_total_next_frag_ind = total_next_frag_ind[train_num:]

    train_total_cur_shape = total_cur_shape[:train_num]
    test_total_cur_shape = total_cur_shape[train_num:]

    train_total_poly_centered = total_poly_centered[:train_num]
    test_total_poly_centered = total_poly_centered[train_num:]

    train_total_next_frag_pos = total_next_frag_pos[:train_num]
    test_total_next_frag_pos = total_next_frag_pos[train_num:]

    train_total_angle = total_angle[:train_num]
    test_total_angle = total_angle[train_num:]

    train_label = label_data[:train_num]
    test_label = label_data[train_num:]

    np.save(os.path.join(train_data_path, 'train_total_next_frag.npy'), train_total_next_frag)
    np.save(os.path.join(test_data_path, 'test_total_next_frag.npy'), test_total_next_frag)

    np.save(os.path.join(train_data_path, 'train_total_next_frag_ind.npy'), train_total_next_frag_ind)
    np.save(os.path.join(test_data_path, 'test_total_next_frag_ind.npy'), test_total_next_frag_ind)

    np.save(os.path.join(train_data_path, 'train_total_cur_shape.npy'), train_total_cur_shape)
    np.save(os.path.join(test_data_path, 'test_total_cur_shape.npy'), test_total_cur_shape)

    np.save(os.path.join(train_data_path, 'train_total_next_frag_pos.npy'), train_total_next_frag_pos)
    np.save(os.path.join(test_data_path, 'test_total_next_frag_pos.npy'), test_total_next_frag_pos)

    np.save(os.path.join(train_data_path, 'train_total_angle.npy'), train_total_angle)
    np.save(os.path.join(test_data_path, 'test_total_angle.npy'), test_total_angle)

    np.save(os.path.join(train_data_path, 'train_label.npy'), train_label)
    np.save(os.path.join(test_data_path, 'test_label.npy'), test_label)

    pickle.dump(train_total_poly_centered, open(os.path.join(train_data_path, 'train_frag.pt'), 'wb'))
    pickle.dump(test_total_poly_centered, open(os.path.join(test_data_path, 'test_frag.pt'), 'wb'))
