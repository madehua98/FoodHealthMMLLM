from utils.file_utils import *

food_image_roots = [
    '/home/xuzhenbo/dishes_wFG/plate_seg_all',
    '/home/xuzhenbo/dishes_DaNeng',
    '/home/xuzhenbo/dishes_DaNeng_val/',
    '/home/xuzhenbo/dishes_wFG/plate_seg_relabelled_230722',
    '/home/xuzhenbo/dishes_wFG/android_bad_1227',
    '/home/xuzhenbo/food_dataset/isia_200',
    '/home/xuzhenbo/food_dataset/isia_500',
    '/home/xuzhenbo/food_dataset/food-101',
    '/home/xuzhenbo/food_dataset/Food2k_complete',
    '/home/xuzhenbo/fresh_data/生鲜基础库226种_230614',
    '/home/xuzhenbo/food_dataset/eating_scenario',
    '/home/xuzhenbo/food_dataset/eating_scenario',
               ]

nonfood_image_roots = [
    '/home/xuzhenbo/food_dataset/nonfood_Stanford_Online_Products',
    # '/home/xuzhenbo/food_dataset/product10k',
    # '/home/xuzhenbo/food_dataset/异物_crops',
    '/home/xuzhenbo/food_dataset/COCO_train2017_nofood'
]

train_info, val_info = {}, {}
for image_label, image_roots, max_samples in zip([0, 1], [nonfood_image_roots, food_image_roots], [100000, 30000]):
    for iid, image_root in enumerate(image_roots):
        im_paths = make_dataset(image_root, suffix=['.jpg', '.JPG'])
        if 'nonfood' in image_root:
            im_paths += make_dataset(image_root, suffix='.crop')
        random.shuffle(im_paths)
        sel_im_paths = im_paths[:max_samples]
        train_sample_count = int(len(sel_im_paths) * 0.8)
        # if not(image_label == 0 or iid<5):
        #     for im_path in sel_im_paths:
        #         remove_exif(im_path)

        for im_path in sel_im_paths[:train_sample_count]:
            train_info[im_path] = image_label
        for im_path in sel_im_paths[train_sample_count:]:
            val_info[im_path] = image_label
        print(image_root, len(sel_im_paths))
print(len(train_info), len(val_info))

all_im_paths = list(train_info.keys()) + list(val_info.keys())
run_imap_multiprocessing(remove_exif, all_im_paths, 64)
print('remove_exif finished')
save_pickle('/home/xuzhenbo/food_dataset/hasFood_samples.pkl', {'train':train_info, 'val':val_info})


