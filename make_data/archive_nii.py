import os
import pandas as pd
import numpy as np
import json
import sys
from utils import load_json_config, ArchiveUtils


if __name__ == '__main__':
    archive_utils = ArchiveUtils(load_json_config(sys.argv[1]))
    archive_utils.archive_dcm()
    archive_utils.convert_nii()
    # archive_utils.add_inclusion_flag()
    archive_utils.generate_hierarchy_preview_png()

