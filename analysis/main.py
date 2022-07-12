import os
# from config.definitions import ROOT_DIR
import bb_analysis as ban 
import numpy as np
import img_cutter as icut
import glob

output_dict = {'num_detections': 100, 'detection_boxes': np.array([[0.39286372, 0.9050959 , 0.51845455, 0.93063146],
       [0.43714142, 0.12623109, 0.54556054, 0.1489396 ],
       [0.4081627 , 0.45146   , 0.51207894, 0.47085685],
       [0.3934756 , 0.6122947 , 0.76804835, 0.70105267],
       [0.41654897, 0.55210984, 0.734138  , 0.61037326],
       [0.39125475, 0.85445356, 0.49232677, 0.8749292 ],
       [0.41502598, 0.42332226, 0.5037766 , 0.440758  ],
       [0.42158055, 0.45642313, 0.53135437, 0.47703448],
       [0.44331357, 0.1016591 , 0.5561815 , 0.12693015],
       [0.4036347 , 0.50090796, 0.49868938, 0.5197144 ],
       [0.4434473 , 0.5701478 , 0.7668275 , 0.6415482 ],
       [0.13182518, 0.21149056, 0.30142286, 0.27049276],
       [0.69994485, 0.22323672, 1.        , 0.34635973],
       [0.8118274 , 0.25563473, 0.9239423 , 0.32170033],
       [0.4464069 , 0.6427327 , 0.7571781 , 0.712514  ],
       [0.5083143 , 0.00278594, 0.77762264, 0.01359119],
       [0.8388346 , 0.33786058, 1.        , 0.46334517],
       [0.7919277 , 0.16323023, 0.9428339 , 0.42385346],
       [0.06359135, 0.15497927, 0.20727946, 0.42726308],
       [0.01185727, 0.6068151 , 0.5492236 , 1.        ],
       [0.        , 0.05858092, 0.19373906, 0.20079868],
       [0.69670415, 0.4294873 , 1.        , 0.55744874],
       [0.43068233, 0.55883336, 0.61518514, 0.58228326],
       [0.00755718, 0.05466607, 0.15370598, 0.31422246],
       [0.5087893 , 0.58304816, 0.7264805 , 0.6208779 ],
       [0.        , 0.88530236, 0.07154369, 0.96977216],
       [0.44538206, 0.00213247, 0.9410091 , 0.03112954],
       [0.42530566, 0.4173376 , 0.5248444 , 0.43730134],
       [0.49972066, 0.5527425 , 0.7063323 , 0.5848813 ],
       [0.02928535, 0.1469879 , 0.1490817 , 0.216767  ],
       [0.86047876, 0.24859802, 0.9774687 , 0.32087052],
       [0.0337561 , 0.37361184, 0.15490343, 0.43778315],
       [0.6971375 , 0.27746174, 1.        , 0.3954802 ],
       [0.3706489 , 0.57987005, 1.        , 0.8351533 ],
       [0.9386133 , 0.5711817 , 1.        , 0.6409326 ],
       [0.08209915, 0.26379493, 0.20417877, 0.33271483],
       [0.69136506, 0.3823011 , 1.        , 0.50728977],
       [0.        , 0.36068496, 0.75042146, 0.6313245 ],
       [0.7506331 , 0.16669357, 1.        , 0.30325496],
       [0.        , 0.10977686, 0.2513903 , 0.25505376],
       [0.9291718 , 0.37043944, 1.        , 0.44141176],
       [0.46068546, 0.5452523 , 0.6699563 , 0.57334507],
       [0.0311301 , 0.31171963, 0.14872372, 0.38399407],
       [0.9390469 , 0.6930014 , 0.9926452 , 0.750963  ],
       [0.00831458, 0.21369798, 0.15267575, 0.48183995],
       [0.5456844 , 0.5286268 , 0.8582217 , 1.        ],
       [0.11673015, 0.18714598, 0.38477206, 0.28989   ],
       [0.8099544 , 0.20651178, 0.9238436 , 0.27494082],
       [0.8471437 , 0.23229311, 1.        , 0.3565172 ],
       [0.36023924, 0.9660374 , 0.8805759 , 0.9916887 ],
       [0.        , 0.01529558, 0.2359336 , 0.14507069],
       [0.05854198, 0.05570154, 0.20965156, 0.31762588],
       [0.4561868 , 0.        , 0.65361285, 0.39844394],
       [0.94171786, 0.63317287, 0.99901044, 0.69697857],
       [0.50105643, 0.52381015, 1.        , 1.        ],
       [0.02698798, 0.25537026, 0.14839604, 0.3288629 ],
       [0.15768944, 0.2074376 , 0.5319182 , 0.8784592 ],
       [0.841851  , 0.39175305, 1.        , 0.51415217],
       [0.        , 0.00630452, 0.09447699, 0.254314  ],
       [0.        , 0.77249587, 0.7339543 , 0.9910438 ],
       [0.44500777, 0.20623443, 0.75846314, 0.27959317],
       [0.9063448 , 0.35944974, 1.        , 0.6473032 ],
       [0.6328964 , 0.04569122, 1.        , 0.53393173],
       [0.9342515 , 0.5271298 , 1.        , 0.5969501 ],
       [0.43845263, 0.        , 1.        , 0.4116755 ],
       [0.3095097 , 0.        , 0.80219334, 0.2375612 ],
       [0.        , 0.22143513, 0.6285097 , 0.7373816 ],
       [0.13469872, 0.04186057, 0.25432044, 0.11190803],
       [0.8434767 , 0.10649067, 0.99769205, 0.38066813],
       [0.        , 0.77042294, 0.22706842, 1.        ],
       [0.90864325, 0.55322707, 1.        , 0.86249864],
       [0.8441474 , 0.43994224, 1.        , 0.56717515],
       [0.        , 0.42604792, 0.60401404, 0.9503956 ],
       [0.32812995, 0.5312867 , 0.66131383, 1.        ],
       [0.        , 0.79129964, 0.09454185, 1.        ],
       [0.6524226 , 0.00834043, 1.        , 0.147619  ],
       [0.69521654, 0.33475238, 1.        , 0.45449567],
       [0.4795342 , 0.20037478, 0.9653273 , 0.8625873 ],
       [0.8327657 , 0.532023  , 1.        , 0.9321477 ],
       [0.        , 0.        , 0.32349655, 0.07955946],
       [0.10527329, 0.        , 0.25992942, 0.20085229],
       [0.5195989 , 0.30786502, 1.        , 0.67903227],
       [0.        , 0.2259267 , 0.30336052, 0.35171425],
       [0.58127165, 0.16742937, 0.73751676, 0.21461858],
       [0.44559368, 0.7262438 , 0.6830912 , 1.        ],
       [0.23511934, 0.789995  , 1.        , 1.        ],
       [0.8522912 , 0.04155168, 0.9807111 , 0.11300965],
       [0.        , 0.04216059, 0.55156964, 0.2403614 ],
       [0.9046407 , 0.45739573, 1.        , 0.7523963 ],
       [0.866283  , 0.2051472 , 0.9711948 , 0.27402127],
       [0.36194298, 0.18664312, 0.7017044 , 0.8199618 ],
       [0.13861018, 0.13959397, 0.44398803, 0.2675578 ],
       [0.18298289, 0.5316294 , 0.81002045, 0.8649511 ],
       [0.8455081 , 0.2846276 , 1.        , 0.40913615],
       [0.2499092 , 0.37558323, 0.4530865 , 0.9395123 ],
       [0.        , 0.        , 0.40001392, 0.17340976],
       [0.2561504 , 0.7403813 , 0.44405714, 1.        ],
       [0.        , 0.        , 0.27287325, 0.26043165],
       [0.        , 0.1635905 , 0.18280178, 0.31691414],
       [0.82445115, 0.49545205, 1.        , 0.62327766]], dtype=np.float32), 'detection_scores': np.array([0.9985323 , 0.9833804 , 0.98168993, 0.7849078 , 0.75645816,
       0.5765627 , 0.33217776, 0.12111115, 0.02024564, 0.01233277,
       0.0099704 , 0.00592202, 0.00581479, 0.00535125, 0.00498128,
       0.00490838, 0.00480935, 0.00472489, 0.00464937, 0.00449118,
       0.00441742, 0.00425357, 0.00421196, 0.00418925, 0.00402638,
       0.00398996, 0.00395811, 0.00395411, 0.00389576, 0.00386983,
       0.00386402, 0.00375757, 0.00369349, 0.00367856, 0.00355333,
       0.00353879, 0.0034875 , 0.00346714, 0.00345457, 0.00344136,
       0.00341704, 0.0033851 , 0.00330174, 0.00330067, 0.00326064,
       0.0032571 , 0.00323507, 0.00322872, 0.00322473, 0.00321394,
       0.00320435, 0.00316235, 0.00316158, 0.00315961, 0.00312442,
       0.00309324, 0.00307232, 0.00303411, 0.00303236, 0.0030148 ,
       0.00298259, 0.0029794 , 0.00296378, 0.00291762, 0.00287935,
       0.00286114, 0.00286064, 0.00286001, 0.00285462, 0.00284967,
       0.00284916, 0.00284827, 0.00282884, 0.00282192, 0.00281936,
       0.00281098, 0.00280952, 0.00275314, 0.00275072, 0.00271294,
       0.00269553, 0.00267717, 0.00267383, 0.00266707, 0.00266171,
       0.00265691, 0.0026536 , 0.00264737, 0.00264436, 0.00262198,
       0.00260499, 0.00260285, 0.00260121, 0.00259393, 0.00258356,
       0.00256997, 0.00255951, 0.00253803, 0.00251868, 0.00248921],
      dtype=np.float32), 'detection_classes': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)}

ROOT_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), '..'))

test_folder = os.path.join(ROOT_DIR, 'test')
csv_folder = os.path.join(ROOT_DIR, 'bb_savings')
image_file = os.path.join(test_folder, 'c1s2_072496.jpg')
image_folder = os.path.join(ROOT_DIR, 'bounding_boxes')
test_label = os.path.join(ROOT_DIR, 'test_labels.csv')
gt_bb = os.path.join(ROOT_DIR, 'gt_bounding_boxes/')

imgs = glob.glob(os.path.join(test_folder, "*.jpg"))

# print(ban.bounding_box_savings(output_dict, test_folder, csv_folder, image_file, image_folder))

for img in imgs:
       icut.retrieve_gtbb(test_label, img, gt_bb)

# ban.bounding_box_savings(output_dict, test_label, csv_folder, image_file, image_folder)