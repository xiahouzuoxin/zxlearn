import argparse
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .dataset import DataFrameDataset

serving_models = {
    # 'name': {'path': 'path/to/model', 'model': None}
}

class ServeModel:
    def __init__(self, model):
        '''
        model: str or torch.nn.Module object.
            The model should be saved by `torch.save(model, 'path/to/model')`, including both state_dict and model structure.
        '''
        if isinstance(model, str):
            self.model = torch.load(model)
            if model.endswith('.ckpt'):
                self.model = self.model['model']
        else:
            self.model = model
        self.model.eval()

        self.feat_configs = self.model.feat_configs
        self.ds_generator = DataFrameDataset

    def predict(self, df):
        ds = self.ds_generator(df, self.feat_configs, target_cols=None, is_raw=True, is_train=False, n_jobs=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=len(df), shuffle=False, collate_fn=self.ds_generator.collate_fn)
        preds = []
        for batch in loader:
            pred = self.model(batch)
            preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return {'prediction': preds.tolist()}

class ServeRequest(BaseModel):
    '''
    features: dict of features. For example, {'user_id': [1, 2, 3], 'item_id': [1, 2, 3]}
    '''
    features: dict

app = FastAPI()

@app.on_event('startup')
async def init_models():
    for name, model in serving_models.items():
        serving_models[name]['model'] = ServeModel(model['path'])
        print(f'Model {name} loaded from {model["path"]} successfully')

@app.post('/{name}/predict')
async def predict(name: str, req: ServeRequest):
    '''
    name: str, model name
    req: ServeRequest, request object. For example, {'features': {'user_id': [1, 2, 3], 'item_id': [1, 2, 3]}}
    '''
    try:
        if name not in serving_models:
            return {'error': 'Model not found'}
        model = serving_models[name]['model']
        df = pd.DataFrame(req.features)
        return model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/{name}/health')
async def health(name: str):
    '''
    Check the health of the model.
    '''
    if name not in serving_models:
        return {'error': 'Model not found'}
    return {'status': 'ok'}

@app.get('/health')
async def health():
    '''
    Check the health of the server.
    '''
    return {'status': 'ok'}

def test_predict(df, name):
    ''' 
    Test the prediction of the model.
    First launch the server by `python -m core.serve --name {name} --path path/to/model`
    '''
    import requests
    import json

    data = {
        'features': df.to_dict(orient='list')
    }
    print(f"Data: {data}")
    data_json = json.dumps(data)
    response = requests.post(
        f"http://localhost:8000/{name}/predict", 
        data=data_json, 
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        prediction = response.json()
        print(f"Prediction: {prediction}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--port', type=int, default=8000)
    
    args = parser.parse_args()
    serving_models[args.name] = {'path': args.path}

    uvicorn.run(app, host='0.0.0.0', port=args.port)
