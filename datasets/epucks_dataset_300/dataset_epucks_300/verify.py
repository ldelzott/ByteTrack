"""
The following script will open each .JSON file in the current folder and update the label field. A new .JSON file is then created and saved into /outputfoler
"""

import os
import json

os.chdir('/home/dellou/PycharmProjects/LabelConversion/e_pucks_dataset_300/save/outputfolder')
i=0
for fp in os.listdir('.'):
    if fp.endswith('.json'):
        i=i+1
        with open(fp) as json_file:
            data = json.load(json_file)
            for shape in data["shapes"]:
                print(i)
                print(shape["label"])

