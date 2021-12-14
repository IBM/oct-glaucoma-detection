"""
View demographics (age, race, iop, ...) for sample data.
"""

import os.path as osp
import numpy as np
import pandas as pd

from nutsflow import *
from nutsml import *
from common import read_samples


def read_table(filename):
    path = osp.join(r'c:\Maet\Data\NYU\longitudinal', filename)
    return pd.read_excel(path)


def get_samples():
    get_dx = MapCol(1, lambda dx: 'POAG' if dx else 'Normal')
    return read_samples() >> get_dx >> GetCols((2, 1)) >> Collect()


def get_sids(samples):
    get_sid = lambda uid: int(uid.split('-')[0])
    return {get_sid(uid): dx for uid, dx in samples}


def get_uids(samples):
    return {uid: dx for uid, dx in samples}


def add_dx(df, id2dx):
    dxs = df.apply(lambda row: id2dx.get(row[0], np.NaN), axis=1)
    df['dx'] = dxs
    return df.dropna()


def get_gender_race(sid2dx):
    df = read_table('subjects.xlsx')
    return add_dx(df, sid2dx)


def get_age(uid2dx):
    df = read_table('cirrus_onh.xlsx')
    df = df[['uid', 'age_at_visit_date']].drop_duplicates()
    return add_dx(df, uid2dx)


def get_eye(uid2dx):
    df = read_table('cirrus_onh.xlsx')
    df = df[['uid', 'eye']].drop_duplicates()
    return add_dx(df, uid2dx)


def get_iop(uid2dx):
    df = read_table('ocular_exam.xlsx')
    df = df[['uid', 'iop']].dropna()
    df = add_dx(df, uid2dx)
    df['iop'] = pd.to_numeric(df['iop'], errors='coerce')
    df = df.dropna()
    return df


def get_md_ght(uid2dx):
    df = read_table('vf.xlsx')
    df = df[df['qualified'] == 'Y']
    df = df[['uid', 'md', 'ght']].dropna()
    dxs = df.apply(lambda row: uid2dx.get(row[0], np.NaN), axis=1)
    df['dx'] = dxs
    return df.dropna()


def get_counts(df, cols):
    return df[cols].groupby(cols[1:]).agg(['count'])


def get_means(df, cols):
    aggs = ['count', 'mean', 'median', 'std', 'min', 'max']
    return df[cols].groupby(cols[1:]).agg(aggs)


def view_data():
    samples = get_samples()
    sid2dx = get_sids(samples)
    uid2dx = get_uids(samples)

    print('#patients', sid2dx >> Count())
    print('#eyes', uid2dx >> Count())

    df = get_eye(uid2dx)
    print(get_counts(df, ['eye', 'dx']))

    df = get_gender_race(sid2dx)
    print(get_counts(df, ['subject_id', 'race', 'dx']))
    print(get_counts(df, ['subject_id', 'gender', 'dx']))

    df = get_age(uid2dx)
    print(get_means(df, ['age_at_visit_date', 'dx']))

    df = get_iop(uid2dx)
    print(get_means(df, ['iop', 'dx']))

    df = get_md_ght(uid2dx)
    print(get_means(df, ['md', 'dx']))
    print(get_means(df, ['ght', 'dx']))


if __name__ == "__main__":
    view_data()
