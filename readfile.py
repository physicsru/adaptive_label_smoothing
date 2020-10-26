import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image

def find_classes(dir: str)-> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances


classes, class_to_idx = find_classes('/home/super/code/DSCMR/data/pascal/image')
samples_img = make_dataset('/home/super/code/DSCMR/data/pascal/image', class_to_idx)
samples_txt = make_dataset('/home/super/code/DSCMR/data/pascal/text', class_to_idx)

print(classes, class_to_idx)
print(samples_img[:10])
print(samples_txt[:10])
