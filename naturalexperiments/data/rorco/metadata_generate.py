# Generate croissant metadata for the RORCO dataset

# Based on the minimal example at https://github.com/mlcommons/croissant?tab=readme-ov-file#simple-format-example

import pandas as pd

filename = 'rorco_data.csv'

data = pd.read_csv(filename)

json_dict = {
    "@type": "sc:Dataset",
    "name": "RORCO Dataset",
    "description": "A dataset based on the RORCO natural experiment.",
    "license": "https://github.com/rtealwitter/naturalexperiments/blob/main/LICENSE",
    "url": "https://github.com/rtealwitter/naturalexperiments",
}

distribution = {
    "@type" : "Cr:FileObject",
    "@id" : "rorco_data.csv",
    "name" : "rorco_data.csv",
    "contentUrl" : "https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/rorco/rorco_data.csv",
    "encodingFormat" : "text/csv",
    "dateModified" : "2024-06-03",
}

json_dict["distribution"] = [distribution]

fields = []

for i, col in enumerate(data.columns):
    fields.append({
        "@type": "sc:Dataset",
        "name": col,
        "description": "A field in the RORCO dataset.",
        "dataType": data.dtypes[col].name,
        "references" : {
            "@id" : "rorco_data.csv",
        }
    })

json_dict["recordSet"] = {
    '@type': 'sc:RecordSet',
    'name' : 'RORCO Column Names',
    'description' : 'The columns in the RORCO dataset.',
    "field" : fields
}

import json

with open('metadata.json', 'w', encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)