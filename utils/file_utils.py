# -*- coding:utf-8 -*-
"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import sys
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]


plateform_name = get_platform()
if plateform_name == 'OS X':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import pickle
import gzip
import json
import os
import sys
import torch
import shutil
import random
import string
import zipfile
import time
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
try:
    import nvidia_smi
except:
    print('import nvidia_smi failed')

def remove_exif(image_name):
    try:
        image = Image.open(image_name).convert('RGB')
    except:
        print('Invalid, image', image_name)
        return
    if not image.getexif():
        return
    print('removing EXIF from', image_name, '...')
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    image_without_exif.save(image_name)

def get_memory_info(ind=0):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(ind)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print("Total memory:", info.total)
    # print("Free memory:", info.free)
    # print("Used memory:", info.used)
    nvidia_smi.nvmlShutdown()
    return info.used

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def print_tensor_shape(ttt):
    if not isinstance(ttt, tuple):
        ttt = (ttt,)
    for el in ttt:
        print(el.shape)


def run_imap_multiprocessing(func, argument_list, num_processes):
    from multiprocessing import Pool
    from tqdm import tqdm
    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm


def get_csv_content(csv_path, delimiter=',', remove_ind=True):
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        content = []
        for row in enumerate(spamreader):
            content.append(row if not remove_ind else row[1])
        return content


def write_rows_to_csv(csv_file_path, rows):
    assert isinstance(rows, list)
    # open the file in the write mode
    f = open(csv_file_path, 'w')
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    for row in rows:
        writer.writerow(row)
    # close the file
    f.close()


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)
    return result_str


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def read_profile(model=None, input=None):
    from thop import profile, clever_format
    if model == None:
        from torchvision.models import resnet50
        model = resnet50()
    if input == None:
        input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def remove_key_word(previous_dict, keywords):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    removed_keys = []
    for k, v in previous_dict.items():
        if_exist_keyword = [1 if el in k else 0 for el in keywords]
        if sum(if_exist_keyword) == 0:
            new_state_dict[k] = v
        else:
            removed_keys.append(k)
    print(f'remove {keywords}: {removed_keys}')
    return new_state_dict


def load_weights_from_data_parallel(model_path, net):
    if not os.path.isfile(model_path):
        print('%s not found' % model_path)
        exit(0)
    else:
        print('Load from %s' % model_path)
    previous_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in previous_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=True)
    return net


def remove_module_in_dict(loaded_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    omitted_keys = []
    for k, v in loaded_dict.items():
        # name = k[7:]  # remove `module.`
        try:
            if k.split('.', 1)[0] == 'model' or k.split('.', 1)[0] == 'backbone' or k.split('.', 1)[0] == 'module':
                name = k.split('.', 1)[1]
                new_state_dict[name] = v
            else:
                # print(k, 'omitted')
                omitted_keys.append(k)
        except:
            print(k, 'convert fail')
    print('Omitted num:', len(omitted_keys), 'Omitted key samples:',omitted_keys[:10])
    return new_state_dict


def load_weights(model_path, net, strict=True):
    if not os.path.isfile(model_path):
        print('%s not found' % model_path)
        exit(0)
    else:
        print('Load from %s' % model_path)
    previous_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(previous_dict, strict=strict)
    return net


def remove_and_mkdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(path, 'removed')
    os.makedirs(path)


def mkdir_if_no(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)
    return


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object


def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def save_pickle2(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def save_pickle4(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_json(filename):
    return json.load(open(filename, 'r'))


def save_json(filename, res):
    json.dump(res, open(filename, 'w'), indent=4)


def save_json_with_np(filename, res):
    import numpy as np
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    json.dump(res, open(filename, 'w'), cls=NpEncoder)


def is_image_file(filename, suffix=None):
    if suffix is not None:
        IMG_EXTENSIONS = [suffix] if not isinstance(suffix, list) else suffix
    else:
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
        ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, suffix=None, max=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, suffix):
                path = os.path.join(root, fname)
                images.append(path)
    if max is not None:
        return images[:max]
    else:
        return images


def make_dataset_prefix(dir, prefix):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.startswith(prefix):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def write_ply(filename, points=None, mesh=None, as_text=False):
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    Returns
    -------
    boolean
        True if no problems
    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True


def write_to_file(dst_path, upload_content):
    with open(dst_path, "w") as f:
        for line in upload_content:
            print(line, file=f)


class obj:
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)


def getSubDirs(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def getSubSubDirs(path):
    subdirs = getSubDirs(path)
    subsubdirs = []
    for subdir in subdirs:
        subsubdirs += getSubDirs(subdir)
    return subsubdirs


def getSubDirNames(srcRoot):
    return [el for el in os.listdir(srcRoot) if os.path.isdir(os.path.join(srcRoot, el))]


def unzip_with_timestamp(inFile, outDirectory):
    # outDirectory = 'C:\\TEMP\\'
    # inFile = 'test.zip'
    fh = open(os.path.join(outDirectory, inFile), 'rb')
    z = zipfile.ZipFile(fh)

    for f in z.infolist():
        name, date_time = f.filename, f.date_time
        name = os.path.join(outDirectory, name)
        with open(name, 'wb') as outFile:
            outFile.write(z.open(f).read())
        date_time = time.mktime(date_time + (0, 0, -1))
        os.utime(name, (date_time, date_time))


import csv


def get_csv_content(csv_path, delimiter=',', remove_ind=True):
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        content = []
        for row in enumerate(spamreader):
            content.append(row if not remove_ind else row[1])
        return content


