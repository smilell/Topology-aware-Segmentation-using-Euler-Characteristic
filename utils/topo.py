import gudhi as gd
import numpy as np

def pre_process(info):
    '''info in shape [dim, b, d, b_x, b_y. b_z, d_x, d_y, d_z]'''

    revised_row_array = np.where(info[:, 2] > 1)[0]
    for row in revised_row_array:
        info[row, 2] = 1

    dim_eff_all = np.unique(info[:, 0])
    pd_gt_1 = {}
    bcp_gt_1 = {}
    dcp_gt_1 = {}

    for dim_eff in dim_eff_all:
        idx = info[:, 0] == dim_eff
        pd_gt_1.update({str(int(dim_eff)): info[idx][:, 1:3]})
        bcp_gt_1.update({str(int(dim_eff)): info[idx][:, 3:6]})
        dcp_gt_1.update({str(int(dim_eff)): info[idx][:, 6::]})

    return pd_gt_1, bcp_gt_1, dcp_gt_1

def get_betti(info):
    betti = np.zeros((3,))
    pd_lh, bcp_lh, dcp_lh = pre_process(info)
    for i in range(3):
        betti[i] = len(pd_lh[str(i)]) if str(i) in list(pd_lh.keys()) else 0
    return betti


def reidx_f_gudhi(idx, ori_shape):
    re_idx = np.zeros(3, dtype=np.uint16)
    reidx_0 = np.array(np.unravel_index(idx, ori_shape, order='C'))
    reidx_1 = np.array(np.unravel_index(idx, ori_shape, order='F'))
    if len(ori_shape)==3:
        div_0 = ori_shape[1] * ori_shape[2]
        re_idx[0] = int(idx //div_0)
        mod_0 = idx % div_0

        re_idx[1] = int(mod_0 // ori_shape[2])          # updated on 23/09/02 from ori_shape[1] to ori_shape[2]
        re_idx[2] = int(mod_0 % ori_shape[2])
        if idx != re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[2] + re_idx[2]:       # updated on 23/09/02 from re_idx[1] * ori_shape[1] to re_idx[1] * ori_shape[2]
            print('hold on, wrong reidx')
        assert (reidx_0 == re_idx).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'

        # div_0 = ori_shape[1] * ori_shape[0]
        # re_idx[2] = int(idx // div_0)
        # mod_0 = idx % div_0
        #
        # re_idx[1] = int(mod_0 // ori_shape[0])
        # re_idx[0] = int(mod_0 % ori_shape[0])
        # assert idx == re_idx[2] * ori_shape[1] * ori_shape[0] + re_idx[1] * ori_shape[0] + re_idx[0]
        # assert (reidx_1 == re_idx).all(), 'the reindex is not following mode F'
        #
        # if not (re_idx[0] <= ori_shape[0] and re_idx[1] <= ori_shape[1] and re_idx[2] <= ori_shape[2]):
        #     print('hi')

    elif len(ori_shape) ==2:
        re_idx[0] = int(idx // ori_shape[1])
        re_idx[1] = int(idx % ori_shape[1])
        assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
        assert (reidx_0 == re_idx[0:2]).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'

        # re_idx[1] = int(idx // ori_shape[0])
        # re_idx[0] = int(idx % ori_shape[0])
        # assert idx == re_idx[1] * ori_shape[0] + re_idx[0]
    return re_idx      #re_idx

def check_point_exist(r_max, p):
    result = True
    for i, point in enumerate(list(p)):
        if point < 0 or point >= r_max[i]:
            result = False
    return result


def get_topo_gudhi(map):
    cc = gd.CubicalComplex(dimensions=map.shape[::-1], top_dimensional_cells=1 - map.flatten())       #
    ph = cc.persistence()
    # betti_2 = cc.persistent_betti_numbers(from_value=1, to_value=0)
    x = cc.cofaces_of_persistence_pairs()

    '''3.1 get birth and death point coordinate from gudhi, and generate info array'''
    info_gudhi = np.zeros((len(ph), 9))
    # x will lack one death point where the filtration is inf
    '''3.1.1 manually write the inf death point'''
    reidx_birth_0 = reidx_f_gudhi(x[1][0][0], map.shape)
    if len(map.shape) == 2:
        birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1]]
    elif len(map.shape) == 3:
        birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2]]

    info_gudhi[0, :] = [0, birth_filtration, 1,
                        reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2],
                        0, 0, 0]
    idx_row = 1
    for dim in range(len(x[0])):
        for idx in range(x[0][dim].shape[0]):
            idx_brith, idx_death = x[0][dim][idx]
            reidx_birth = reidx_f_gudhi(idx_brith, map.shape)
            reidx_death = reidx_f_gudhi(idx_death, map.shape)

            if len(map.shape) == 2:
                if check_point_exist(list(map.shape), reidx_birth[0:2]) and check_point_exist(list(map.shape), reidx_death[0:2]):
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1]]
                else:
                    print('wrong topo loss in topo.py, hold')
            elif len(map.shape) == 3:
                birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1], reidx_birth[2]]
                death_filtration = 1 - map[reidx_death[0], reidx_death[1], reidx_death[2]]
            else:
                assert False, 'wrong input dimension!'

            info_gudhi[idx_row, :] = [dim, birth_filtration, death_filtration,
                                      reidx_birth[0], reidx_birth[1], reidx_birth[2],
                                      reidx_death[0], reidx_death[1], reidx_death[2]]
            idx_row += 1

    betti = get_betti(info_gudhi)
    return betti, info_gudhi