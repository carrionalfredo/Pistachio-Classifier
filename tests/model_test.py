import classify_service

def test_predict():
    model=classify_service.load_mlflow_model()
    features = {
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
    actual_prediction = model.predict(features)
    expected_predict = 0

    assert actual_prediction==expected_predict

