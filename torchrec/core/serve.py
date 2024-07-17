import argparse
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .dataset import DataFrameDataset

serving_models = {
    # 'name': {'path': 'path/to/model', 'model': None, 'batch_size': 8}
}

class ServeModel:
    def __init__(self, model, batch_size=8):
        '''
        model: str or torch.nn.Module object by `torch.save(model, 'path/to/model')`
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
        self.batch_size = batch_size

    def predict(self, features: dict):
        df = pd.DataFrame([features])
        ds = self.ds_generator(df, self.feat_configs, target_cols=None, is_raw=True, is_train=False, n_jobs=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        preds = []
        for batch in loader:
            pred = self.model(batch)
            preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return {'prediction': preds.tolist()}

class ServeRequest(BaseModel):
    features: dict

app = FastAPI()

@app.on_event('startup')
async def init_models():
    for name, model in serving_models.items():
        serving_models[name]['model'] = ServeModel(model['path'], model.get('batch_size', 8))

@app.post('/{name}/predict')
async def predict(name: str, req: ServeRequest):
    try:
        if name not in serving_models:
            return {'error': 'Model not found'}
        model = serving_models[name]['model']
        return model.predict(req.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    serving_models[args.name] = {'path': args.path}
    
    uvicorn.run(app, host='0.0.0.0', port=8000)
