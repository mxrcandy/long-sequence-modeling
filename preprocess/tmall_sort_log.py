from collections import Counter

import numpy as np
import tqdm


def calc_category_distribution():
    train_dataset_path = 'sdata/tmall/train_v1.npy'
    train_dataset = np.load(train_dataset_path, allow_pickle=True)
    all_dataset = train_dataset

    cid_list = []
    for i in tqdm.tqdm(range(len(all_dataset))):
        _, _, cid, _, _, _, = all_dataset[i][:6]
        cid_list.append(cid)
    cid_counter = Counter(cid_list)
    print(cid_counter.most_common(10))


def sanity_check():
    per_user_ts_list = dict()
    dataset_path = '../data/tmall/user_log_sorted.csv'

    cnt = 0
    error_cnt = 0

    with open(dataset_path, 'r') as file:
        for line in file:
            uid, iid, cid, _, _, ts, btype = line.split(',')
            if btype == 'action_type\n' or int(btype) != 0:
                continue

            cnt += 1
            if uid not in per_user_ts_list:
                per_user_ts_list[uid] = []
            if len(per_user_ts_list[uid]) > 0 and int(ts) < per_user_ts_list[uid][-1]:
                error_cnt += 1
            per_user_ts_list[uid].append(int(ts))

            if cnt >= 200000:
                break

    print(error_cnt, cnt)


def sort_log():
    per_user_log = dict()
    dataset_path = 'data/tmall/user_log.csv'

    with open(dataset_path, 'r') as file:
        for line in file:
            uid, iid, cid, _, _, ts, btype = line.split(',')
            if btype == 'action_type\n' or int(btype) != 0 or len(ts) != 4:
                continue
            if uid not in per_user_log:
                per_user_log[uid] = []
            per_user_log[uid].append([line, int(ts)])

    for uid in per_user_log:
        per_user_log[uid].sort(key=lambda x: x[1])
    sorted_log = []
    for uid in per_user_log:
        sorted_log.extend([x[0] for x in per_user_log[uid]])

    save_path = 'data/tmall/user_log_sorted.csv'
    with open(save_path, 'w') as file:
        for line in sorted_log:
            file.write(line)


if __name__ == '__main__':
    sort_log()
