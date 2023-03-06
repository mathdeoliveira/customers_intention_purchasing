import os
import joblib
import pandas as pd
from google.cloud import storage

def load_model(name_file: str):
    """Return model from model directory
    Args:
        name_file: str, name for file to be save
    Returns:
        model loaded from directory
    """
    bucket_name = 'models_customer_intention'
    storage_client = storage.Client('customer-intention')
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(f'{name_file}.joblib')
    folder = '/tmp/'

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    blob.download_to_filename(f'/tmp/{name_file}.joblib')

    model = joblib.load(f'/tmp/{name_file}.joblib')
    return model


def customer_intention(request):
    """
    """
    request_json = request.get_json()

    try:
        pipe = load_model('pipeline')
        model = load_model('final_model')
    except FileNotFoundError as e:
        print(f"The pipeline/model file does not exist: {e}")
        raise e
    data = pd.DataFrame(
        [
            {
                "Administrative": request_json['Administrative'],
                "Administrative_Duration": request_json['Administrative_Duration'],
                "Informational": request_json['Informational'],
                "Informational_Duration": request_json['Informational_Duration'],
                "ProductRelated": request_json['ProductRelated'],
                "ProductRelated_Duration": request_json['ProductRelated_Duration'],
                "BounceRates": request_json['BounceRates'],
                "ExitRates": request_json['ExitRates'],
                "PageValues": request_json['PageValues'],
                "SpecialDay": request_json['SpecialDay'],
                "Month": request_json['Month'],
                "OperatingSystems": request_json['OperatingSystems'],
                "Browser": request_json['Browser'],
                "Region": request_json['Region'],
                "TrafficType": request_json['TrafficType'],
                "VisitorType": request_json['VisitorType'],
                "Weekend": request_json['Weekend'],
            }
        ]
    )

    data_transformed = pipe.transform(data)
    predict = model.predict_proba(data_transformed)

    result = {"pred": predict.tolist()}

    return result
