dataset_info = dict(
    dataset_name='xray',
    paper_info=dict(
        author=' ',
        title=' ',
        container=' ',
        year=' ',
        homepage=' ',
    ),
    keypoint_info={
        0:
        dict(name='c2-0', id=0, color=[0, 0, 255], type='', swap='c2-3'),
        1:
        dict(name='c2-1', id=1, color=[0, 0, 255], type='', swap='c2-2'),
        2:
        dict(name='c2-2', id=2, color=[0, 0, 255], type='', swap='c2-1'),
        3:
        dict(name='c2-3', id=3, color=[0, 0, 255], type='', swap='c2-0'),
        4:
        dict(name='c3-0', id=4, color=[0, 153, 255], type='', swap='c3-3'),
        5:
        dict(name='c3-1', id=5, color=[0, 153, 255], type='', swap='c3-2'),
        6:
        dict(name='c3-2', id=6, color=[0, 153, 255], type='', swap='c3-1'),
        7:
        dict(name='c3-3', id=7, color=[0, 153, 255], type='', swap='c3-0'),
        8:
        dict(name='c4-0', id=8, color=[0, 255, 255], type='', swap='c4-3'),
        9:
        dict(name='c4-1', id=9, color=[0, 255, 255], type='', swap='c4-2'),
        10:
        dict(name='c4-2', id=10, color=[0, 255, 255], type='', swap='c4-1'),
        11:
        dict(name='c4-3', id=11, color=[0, 255, 255], type='', swap='c4-0'),
        12:
        dict(name='c5-0', id=12, color=[0, 255, 0], type='', swap='c5-3'),
        13:
        dict(name='c5-1', id=13, color=[0, 255, 0], type='', swap='c5-2'),
        14:
        dict(name='c5-2', id=14, color=[0, 255, 0], type='', swap='c5-1'),
        15:
        dict(name='c5-3', id=15, color=[0, 255, 0], type='', swap='c5-0'),
        16:
        dict(name='c6-0', id=16, color=[255, 255, 0], type='', swap='c6-3'),
        17:
        dict(name='c6-1', id=17, color=[255, 255, 0], type='', swap='c6-2'),
        18:
        dict(name='c6-2', id=18, color=[255, 255, 0], type='', swap='c6-1'),
        19:
        dict(name='c6-3', id=19, color=[255, 255, 0], type='', swap='c6-0'),
        20:
        dict(name='c7-0', id=20, color=[255, 0, 0], type='', swap='c7-3'),
        21:
        dict(name='c7-1', id=21, color=[255, 0, 0], type='', swap='c7-2'),
        22:
        dict(name='c7-2', id=22, color=[255, 0, 0], type='', swap='c7-1'),
        23:
        dict(name='c7-3', id=23, color=[255, 0, 0], type='', swap='c7-0')},
    skeleton_info={
        0:
        dict(link=('c2-0', 'c2-1'), id=0, color=[0, 0, 255]),
        1:
        dict(link=('c2-1', 'c2-2'), id=1, color=[0, 0, 255]),
        2:
        dict(link=('c2-2', 'c2-3'), id=2, color=[0, 0, 255]),
        3:
        dict(link=('c2-3', 'c2-0'), id=3, color=[0, 0, 255]),
        4:
        dict(link=('c3-0', 'c3-1'), id=4, color=[0, 153, 255]),
        5:
        dict(link=('c3-1', 'c3-2'), id=5, color=[0, 153, 255]),
        6:
        dict(link=('c3-2', 'c3-3'), id=6, color=[0, 153, 255]),
        7:
        dict(link=('c3-3', 'c3-0'), id=7, color=[0, 153, 255]),
        8:
        dict(link=('c4-0', 'c4-1'), id=8, color=[0, 255, 255]),
        9:
        dict(link=('c4-1', 'c4-2'), id=9, color=[0, 255, 255]),
        10:
        dict(link=('c4-2', 'c4-3'), id=10, color=[0, 255, 255]),
        11:
        dict(link=('c4-3', 'c4-0'), id=11, color=[0, 255, 255]),
        12:
        dict(link=('c5-0', 'c5-1'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('c5-1', 'c5-2'), id=13, color=[0, 255, 0]),
        14:
        dict(link=('c5-2', 'c5-3'), id=14, color=[0, 255, 0]),
        15:
        dict(link=('c5-3', 'c5-0'), id=15, color=[0, 255, 0]),
        16:
        dict(link=('c6-0', 'c6-1'), id=16, color=[255, 255, 0]),
        17:
        dict(link=('c6-1', 'c6-2'), id=17, color=[255, 255, 0]),
        18:
        dict(link=('c6-2', 'c6-3'), id=18, color=[255, 255, 0]),
        19:
        dict(link=('c6-3', 'c6-0'), id=19, color=[255, 255, 0]),
        20:
        dict(link=('c7-0', 'c7-1'), id=20, color=[255, 0, 0]),
        21:
        dict(link=('c7-1', 'c7-2'), id=21, color=[255, 0, 0]),
        22:
        dict(link=('c7-2', 'c7-3'), id=22, color=[255, 0, 0]),
        23:
        dict(link=('c7-3', 'c7-0'), id=23, color=[255, 0, 0])
    },
    joint_weights=[1.] * 24,
    sigmas=[])
    # ,
    # stats_info=dict(bbox_center=(528., 427.), bbox_scale=400.))
