import os

import cv2
from joblib import Parallel, delayed
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
from tqdm import tqdm

target_version = 0
n_jobs = 12

target_path = "./data/target_{}.png".format(target_version)
input_dir = "/media/kwaklab_103/sda/data/patch_data/KBSMC_20221224_C03-STO-30"
output_dir = "/media/kwaklab_103/sdc/data/patch_data/KBSMC_20221224_C03-STO-30_normalized_{}".format(target_version)


def make_stain_normalized(source_path, normalizer):
    save_path = source_path.replace(input_dir, output_dir)

    source = cv2.imread(source_path, cv2.IMREAD_COLOR)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    try:
        transformed = normalizer.transform(source)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, transformed)
    except Exception as e:
        print(e)
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        os.makedirs('./data/failed/{}'.format(os.path.basename(output_dir)), exist_ok=True)
        cv2.imwrite('./data/failed/{}/{}'.format(os.path.basename(output_dir), os.path.basename(source_path)), source)
        return


if __name__ == "__main__":
    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    normalizer = MacenkoNormalizer()
    normalizer.fit(target)

    source_paths = []
    for (path, dir, files) in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext not in ('.png', '.jpg', '.jpeg'):
                continue
            source_paths.append(os.path.join(path, filename))

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(make_stain_normalized)(source_path, normalizer) for source_path in tqdm(source_paths))
    else:
        for source_path in tqdm(source_paths):
            make_stain_normalized(source_path, normalizer)
