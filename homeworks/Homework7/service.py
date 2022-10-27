import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
model_runner = model_ref.to_runner()
svc = bentoml.Service("cool_model2", runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(-1,4), enforce_shape=True, dtype=np.float64, enforce_dtype=True), output=NumpyNdarray())
async def classify(vectors):
    prediction = await model_runner.predict.async_run(vectors)
    print(prediction)
    return prediction[0]

