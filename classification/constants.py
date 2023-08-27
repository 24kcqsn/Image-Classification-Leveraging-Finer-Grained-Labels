import numpy as np
# 定义标签和映射
CIFAR20_COARSE = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 'household electrical devices',
    'household furniture', 'insects', 'large carnivores', 'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores', 'medium mammals', 'non-insect invertebrates', 'people', 'reptiles',
    'small mammals',
    'trees', 'vehicles'
]
CIFAR20_LABELS = np.array([4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
            6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
            5, 18,  8,  8, 15, 13, 14, 17, 18, 10,
            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
            10, 3,  2, 12, 12, 16, 12,  1,  9, 18,
            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
            16, 18,  2,  4,  6, 18,  5,  5,  8, 18,
            18,  1,  2, 15,  6,  0, 17,  8, 14, 13
])



