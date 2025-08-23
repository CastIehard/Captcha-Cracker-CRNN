# Information
This README contains general notes regarding the dataset. Here we describe the directory structure and explain formats where we think its required. If you find that some information is unclear you may reach out to us for more information.

# Directory Structure
The directory structure looks like the following
├── ...
├── part2                       # Root directory containing all dataset splits (phase1|2|3 etc.)
│   ├── train                   # Split folder
│   │   ├── images              # Image folder that contains all the captcha images
│   │   │   ├── 000001.png      # Each image in the train folder assumes a unique id (Note: not unique acorss splits)
│   │   │   ├── 000002.png
│   │   │   ├── ...
│   │   │   └── 060000.png
│   │   └── labels.json         # Complete Detectron style JSON file (format details below)
│   ├── test
│   │   └── ...
│   └── val
├── part3
│   ├── test
│   │   ├── images              # Part 3 and 4 will only contain images
│   │   │   └── ...
└── part4

# JSON Label File Format
We provide a json format which combines all labels in a single file. Following is the format for a single entry of the json file associated with the image "019065.png" in the images sub-folder: 
```
[
    {
        "height": 160, 
        "width": 640, 
        "image_id": "019065", 
        "captcha_string": "YC6", 
        "annotations": [
            {
                "bbox": [73.4924, 70.4307, 154.9056, 156.3385], 
                "oriented_bbox": [193.91, 33.40, 160.64, 61.04, 177.12, 80.96, 210.38, 53.32]
                "category_id": 34
            }, ...
        ]
    }
]
```

Note that bounding boxes are provided in absolute terms here i.e. the top left box coordinate followed by the bottom right coordinates: 
[x1, y1, x2, y2]

The "oriented_bbox" field contains the oriented bounding box vertices for the same character in absolute term:
[x1,y1,x2,y2,x3,y3,x4,y4]

In the example above the bounding box annotations are showed for the character Y. The next set of annotations will be for C and then 6.

# Part 4 Distractors List:
For the Part 4 we add the following distractor characters to the test set: "*#?✓"

# Submitting Predictions
You should use the same JSON format as above to create your predictions. Please ensure that you strictly follow the above format with the correct convention otherwise you may loose points during evaluation. Here the captcha_string field is critical.