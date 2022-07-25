import numpy as np
import tensorflow as tf
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Python Implementation')
    parser.add_argument('--num_of_partition', type=int, default=3, help='num_of_partition')
    parser.add_argument('--num_of_data', type=int, default=5, help='num_of_data')
    parser.add_argument('--target_shape', type=str, default='square', help='target_shape')
    parser.add_argument('--dpi', type=int, default=128, help='dpi')
    parser.add_argument('--save_figures', action='store_true', help='save_figures')
    parser.add_argument('--use_random_seq', action='store_true', help='use_random_seq')
    parser.add_argument('--use_rotation', action='store_true', help='use_rotation')
    parser.add_argument('--max_seq', type=int, default=1, help='max_seq')
    parser.add_argument('--split_perpen', action='store_true', help='split_perpen')
    parser.add_argument('--str_datasets', type=str, default='/media/storage/fan/datasets', help='str_datasets')
    parser.add_argument('--target_edge', type=int, default=1, help='target_edge')
    parser.add_argument('--name_of_model', type=str, default='trans_small', help='name_of_model')
    parser.add_argument('--num_epochs', type=int, default=5, help='num_epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--learning_rate_cls', type=float, default=5e-4, help='learning_rate_cls')
    parser.add_argument('--learning_rate_pos', type=float, default=2e-3, help='learning_rate_pos')
    parser.add_argument('--use_early_stop', action='store_true', help='use_early_stop')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='early_stopping_patience')
    parser.add_argument('--padding_rate', type=float, default=0.25, help='padding_rate')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent_dim')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')

    args = parser.parse_args()

    assert args.target_shape in ['square', 'pentagon', 'hexagon', 'octagon']

    return parser, args

def get_str_directory(str_datasets, target_shape, num_of_data, num_of_frags, use_random_seq, split_perpen, use_rotation):
    str_directory = os.path.join(str_datasets, '{}_data_{}_frags_{}'.format(target_shape, num_of_data, num_of_frags))
    if use_random_seq:
        str_directory += '_random'

    if split_perpen:
        str_directory += '_perpen'

    if use_rotation:
        str_directory += '_rotation'

    return str_directory

def create_mask_original(next_frag_ind, num_of_frags):
    mask_y_train = []
    y_train = []

    for i in range(0, next_frag_ind.shape[0], num_of_frags):
        mask_instance = np.ones((num_of_frags, ))
        temp_mask = [mask_instance.copy()]

        for j in range(0, num_of_frags - 1):
            mask_instance[next_frag_ind[i + j]] = 0.0
            temp_mask.append(mask_instance.copy())
            y_train.append(next_frag_ind[i + j])

        temp_mask = np.stack(temp_mask, axis=0)
        mask_y_train.append(temp_mask)
        y_train.append(next_frag_ind[i + 7])

    mask_y_train = np.concatenate(mask_y_train, axis=0).astype(np.float32)
    y_train = np.array(y_train)

    return mask_y_train, y_train

def create_mask_order(y_train, num_of_frags):
    mask_y_train = np.ones((y_train.shape[0], num_of_frags))
    for i in range(0, mask_y_train.shape[0]):
        for j in range(0, num_of_frags):
            if j < y_train[i]:
                mask_y_train[i][j] = 0

    mask_y_train = mask_y_train.astype(np.float32)
    return mask_y_train

def decay_learning_rate(learning_rate_cls, learning_rate_pos):
    learning_rate_cls_ = learning_rate_cls * 0.1
    learning_rate_pos_ = learning_rate_pos * 0.5

    print('learning_rate decayed')
    print('learning_rate_pos =', learning_rate_pos)
    print('learning_rate_cls =', learning_rate_cls)

    return learning_rate_cls_, learning_rate_pos_

def write_ckpt_log_cls_pos(
    template1, template2,
    checkpoint_path,
    ckpt_save_path, epoch,
    train_cls_loss, valid_cls_loss,
    train_pos_loss, valid_pos_loss,
    valid_loss,
    train_cls_accuracy, valid_cls_accuracy,
    train_pos_recall, valid_pos_recall,
    train_pos_dist, valid_pos_dist,
):
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    f = open(os.path.join(checkpoint_path, 'ckpt_log_train.txt'), 'a')
    f.write('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
    f.write('\n')
    f.write(template1.format(
        epoch + 1,
        train_cls_loss.result(), train_pos_loss.result(),
        valid_cls_loss.result(), valid_pos_loss.result(),
        valid_loss.result(),
    ))
    f.write('\n')
    f.write(template2.format(
        epoch + 1,
        train_cls_accuracy.result(), valid_cls_accuracy.result(),
        train_pos_recall.result(), valid_pos_recall.result(),
        train_pos_dist.result(), valid_pos_dist.result(),
    ))
    f.write('\n')
    f.close()

def write_ckpt_log(
    template1, template2,
    checkpoint_path,
    ckpt_save_path, epoch,
    train_cls_loss, valid_cls_loss,
    train_pos_loss, valid_pos_loss,
    train_rot_loss, valid_rot_loss,
    valid_loss,
    train_cls_accuracy, valid_cls_accuracy,
    train_pos_recall, valid_pos_recall,
    train_pos_dist, valid_pos_dist,
    train_rot_acc, valid_rot_acc,
):
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    f = open(os.path.join(checkpoint_path, 'ckpt_log_train.txt'), 'a')
    f.write('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
    f.write('\n')
    f.write(template1.format(
        epoch + 1,
        train_cls_loss.result(), train_pos_loss.result(), train_rot_loss.result(),
        valid_cls_loss.result(), valid_pos_loss.result(), valid_rot_loss.result(),
        valid_loss.result(),
    ))
    f.write('\n')
    f.write(template2.format(
        epoch + 1,
        train_cls_accuracy.result(), valid_cls_accuracy.result(),
        train_pos_recall.result(), valid_pos_recall.result(),
        train_pos_dist.result(), valid_pos_dist.result(),
        train_rot_acc.result(), valid_rot_acc.result(),
    ))
    f.write('\n')
    f.close()

def get_trainable_weights(str_model, str_task, model):
    if str_model == 'trans_small' and str_task == 'cls':
        trainable_weights = model.encoder1.trainable_weights +\
            model.encoder2.trainable_weights +\
            model.denseblock_cls.trainable_weights +\
            model.transformer.trainable_weights +\
            model.dense1.trainable_weights
    elif str_model == 'trans_small' and str_task == 'pos':
        trainable_weights = model.encoder1.trainable_weights +\
            model.encoder2.trainable_weights +\
            model.denseblock_pos.trainable_weights +\
            model.transformer.trainable_weights +\
            model.dense2.trainable_weights +\
            model.decoder.trainable_weights

    return trainable_weights

def compute_coverage(target_shape, cum_image, npad_pixel, dpi):
    target_shape = np.clip(target_shape, 0.0, 1.0)
    cum_image = np.clip(cum_image, 0.0, 1.0)

    target_shape = np.round(target_shape)
    cum_image = np.round(cum_image)

    coverage = np.sum(target_shape * cum_image[npad_pixel:dpi + npad_pixel, npad_pixel:dpi + npad_pixel])
    coverage /= np.sum(target_shape)

    return coverage

def compute_iou(target_shape, cum_image, npad):
    target_shape = np.clip(target_shape, 0.0, 1.0)
    cum_image = np.clip(cum_image, 0.0, 1.0)

    target_shape = np.round(target_shape)
    cum_image = np.round(cum_image)

    padded_target_shape = np.pad(target_shape, npad, 'constant',
        constant_values=(0))

    iou = np.sum(padded_target_shape * cum_image) / np.sum(np.clip(padded_target_shape + cum_image, 0.0, 1.0))
    return iou

def get_target_shape(data_path):
    target_shape = np.load(open(os.path.join(data_path, 'target_shape.npy'), 'rb'), allow_pickle=True)
#    target_shape = np.round(np.clip(target_shape, 0.0, 1.0))
    return target_shape

def get_x_y(data_path, str_phase):
    x = np.load(os.path.join(data_path, '{}_total_cur_plus_frags.npy'.format(str_phase)), allow_pickle=True)
    y = np.load(os.path.join(data_path, '{}_label.npy'.format(str_phase)), allow_pickle=True).astype(np.int32)

#    x = np.round(np.clip(x, 0.0, 1.0))
    x = x[..., np.newaxis]
    return x, y

def get_x_frag_y_pos_angle(data_path, str_phase):
    x_frag = np.load(os.path.join(data_path, '{}_total_next_frag.npy'.format(str_phase)))
    y_pos = np.load(os.path.join(data_path, '{}_total_next_frag_pos.npy'.format(str_phase)))
    angle = np.load(os.path.join(data_path, '{}_total_angle.npy'.format(str_phase)))

#    x_frag = np.round(np.clip(x_frag, 0.0, 1.0))
#    y_pos = np.round(np.clip(y_pos, 0.0, 1.0))

    return x_frag, y_pos, angle

def get_max_indices(pos):
    indices = np.array([
        np.argmax(np.amax(pos, axis=1)),
        np.argmax(np.amax(pos, axis=0)),
    ])
    return indices

def print_info(args):
    print('num_of_partition =', args.num_of_partition)
    print('num_of_data =', args.num_of_data)
    print('num_of_frags =', 2**args.num_of_partition)
    print('target_shape =', args.target_shape)
    print('save_figures =', args.save_figures)
    print('use_random_seq =', args.use_random_seq)
    print('use_rotation =', args.use_rotation)
    print('split_perpen =', args.split_perpen)
    print('max_seq =', args.max_seq)

def get_path_ckpt(str_model, str_target_shape, num_of_frags, latent_dim, batch_size, trans_num_layer, trans_num_head, trans_dff, lr, esp, num_of_data, use_random_seq, split_perpen, use_rotation):
    str_path = '../checkpoints'

    path_ckpt = '{}_ts_{}_nf_{}_ld_{}_bs_{}_trans_{}_{}_{}_lr_{}_esp_{}_nofd_{}'.format(
        str_model, str_target_shape, num_of_frags,
        latent_dim, batch_size, trans_num_layer,
        trans_num_head, trans_dff, lr,
        esp, num_of_data
    )

    if use_random_seq:
        path_ckpt += '_random'

    if split_perpen:
        path_ckpt += '_perpen'

    if use_rotation:
        path_ckpt += '_rotation'

    path_all = os.path.join(str_path, path_ckpt)

    return path_all, path_ckpt
