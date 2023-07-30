import requests


data = {
    "area": 73107,
    "perimeter": 1161.8070,
    "major_axis": 442.4074,
    "minor_axis": 217.7261,
    "eccentricity": 0.8705,
    "eqdiasq": 305.0946,
    "solidity": 0.9424,
    "convex_area": 77579,
    "extent": 0.7710,
    "aspect_ratio": 2.0319,
    "roundness": 0.6806,
    "compactness": 0.6896,
    "shapefactor_1": 0.0061,
    "shapefactor_2": 0.0030,
    "shapefactor_3": 0.4756,
    "shapefactor_4": 0.9664
}

url= 'http://localhost:9696/classify'
response = requests.post(url, json=data)
print(response.json())


data = {
    "area": 96395,
    "perimeter": 1352.6740,
    "major_axis": 515.8730,
    "minor_axis": 246.5945,
    "eccentricity": 0.8784,
    "eqdiasq": 350.3340,
    "solidity": 0.9549,
    "convex_area": 100950,
    "extent": 0.7428,
    "aspect_ratio": 2.0920,
    "roundness": 0.6620,
    "compactness": 0.6791,
    "shapefactor_1": 0.0054,
    "shapefactor_2": 0.0026,
    "shapefactor_3": 0.4612,
    "shapefactor_4": 0.9648
}

url= 'http://localhost:9696/classify'
response = requests.post(url, json=data)
print(response.json())