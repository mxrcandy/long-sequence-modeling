def sort_log():
    per_user_log = dict()
    dataset_path = 'data/alipay/user_log.csv'

    with open(dataset_path, 'r') as file:
        for line in file:
            uid, _, iid, cid, btype, ts = line.split(',')
            ts = ts[:-1]

            if btype == 'act_ID' or int(btype) != 0 or len(ts) != 8:
                continue

            if uid not in per_user_log:
                per_user_log[uid] = []
            per_user_log[uid].append([line, int(ts)])

    for uid in per_user_log:
        per_user_log[uid].sort(key=lambda x: x[1])
    sorted_log = []
    for uid in per_user_log:
        sorted_log.extend([x[0] for x in per_user_log[uid]])

    save_path = 'data/alipay/user_log_sorted.csv'
    with open(save_path, 'w') as file:
        for line in sorted_log:
            file.write(line)


def sanity_check():
    uid_list = set()
    dataset_path = 'data/alipay/user_log.csv'
    with open(dataset_path, 'r') as file:
        for line in file:
            uid, _, iid, cid, btype, ts = line.split(',')
            ts = ts[:-1]

            if btype == 'act_ID' or int(btype) != 0 or len(ts) != 8:
                continue

            uid_list.add(int(uid))

    uid_list = list(uid_list)
    uid_list.sort()
    print(len(uid_list), uid_list[:10], uid_list[-10:])


if __name__ == '__main__':
    # sanity_check()
    sort_log()
