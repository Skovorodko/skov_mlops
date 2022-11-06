import pandas as pd
import pickle
import os
import json
from flask import Flask
from flask_restful import Resource, Api
from flask import Flask, jsonify, abort
from flask_restx import Api, Resource
from flask_restful import reqparse
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor


app = Flask(__name__)
api = Api(app)

model_path = 'models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)


api.parser().add_argument('file_name',
                          location = '/Users/alexanderskovorodko/Desktop/MLops2/model/train.csv',
                          type=str)

api.parser().add_argument('add_example',
                          default = '0',
                          type=str)

api.parser().add_argument('model_name',
                          choices = ['RandomForestClassifier', 'XGBRegressor'],
                          default = "RandomForestClassifier",
                          required=True)

api.parser().add_argument('way_file_json',
                          location = ['/Users/alexanderskovorodko/Desktop/MLops2/json_models/RandomForestClassifier.json',
                                     '/Users/alexanderskovorodko/Desktop/MLops2/json_models/XGBRegressor.json'],
                          default = '/Users/alexanderskovorodko/Desktop/MLops2/json_models/RandomForestClassifier.json',
                          type=str)

models = {
    "RandomForestClassifier": {"min_samples_split": 8, "n_estimators": 200, "min_samples_leaf": 2},
    "XGBRegressor": {"max_depth": 3, "n_estimators": 150, "learning_rate": 0.05}
}

path_to = {
    'RandomForestClassifier': f"{model_path}model_RF.pkl",
    'XGBRegressor': f"{model_path}modelxgb.cbm",
    1: f"{model_path}model_RF.cbm",
    2: f"{model_path}modelxgb.pkl"
}


@api.route('/train', methods = ['PUT'], doc ={'description': 'Train_model'})
@api.expect(api.parser())

class Train(Resource):
    @api.doc(parameters = {'train_name':f'Train dataset'})
    @api.doc(parameters = {'test_name':f'Test dataset'})
    @api.doc(parameters = {'add_example': f'Number of example'})
    @api.doc(parameters = {'model_name': f'Model name'})
    @api.doc(parameters = {'path_file_json': f'File path is: '})

    # successful and fucks up situations (usually fucks up)
    @api.doc(response ={200: 'Success'}) # https://flask-restx.readthedocs.io/en/latest/swagger.html
    @api.doc(response ={403: 'Not Authorized'})
    @api.doc(response ={400: 'Validation Error'})

    def put(self):
        # load model
        a = api.parser().parse_args()
        model = self.load(a.path_file_json)

        # reading files
        train_file = pd.read_csv(a.train_name)
        test_file = pd.read_csv(a.test_name)
        # fit
        model.fit(train_file, test_file)
        # saving
        save = "/results" + a.path_file_json[a.path_file_json.rfind('/')+
             1:a.path_file_json.rfind(
                '_'
            )
        ] + "_" + a.add_example + ".pkl"
        os.makedirs(os.path.dirname(save), exist_ok=True)

        if os.path.isfile(save):
            pickle.dump(model, open(save, 'wb'))
            return {'Model is fitted, but it is exist -> error: ', 202}
        else:
            pickle.dump(model, open(save, 'wb'))
            return 'Model is fitted with success', 200


    # let's download model to our tutorial
    @staticmethod
    def load(_model_path):
        model_paths = os.getcwd() + '/' + _model_path
        with open(model_paths, 'r') as JSON:
            parametrs = json.load(JSON)
        # Upload model by their name
        if "RandomForestClassifier" in _model_path:
            return RandomForestClassifier(**parametrs)
        elif 'XGBRegressor' in _model_path:
            return XGBRegressor(**parametrs)
        else:
            api.abort(403, message = 'Model with that params is non - def')


@api.route('/models', methods = ['GET', 'DELETE'])
@api.expect(api.parser())
class ListOfSavedModels(Resource):
    # all types of models

    def get(self):
        t = {'Saved models': os.listdir(model_path)}
        return jsonify(t)

    @api.doc(parameters = {'add_example': f'Number of example about del'})
    @api.doc(parameters = {'model_name': f'Model name'})

    # delete uniq model
    def del_(self):
        a = api.parser().parse_args()
        deleting_files = "/results" + a.model_name + "_" + a.add_example + ".pkl"
        try:
            os.remove(deleting_files)
        except FileNotFoundError:
            return 'File does not exist', 403
        return 'Model removed', 200

api.add_resource(ListOfSavedModels, '/models')


@api.route('/predict', methods=['POST'])
@api.expect(api.parser())

class Predict(Resource):
    @api.doc(parameters = {'file_name':f'Original dataset'})
    @api.doc(parameters = {'add_example': f'Number of example'})
    @api.doc(parameters = {'model_name': f'Model name'})

    def get(self):
        pass

    def post(self):
        a = api.parser().parse_args()
        # loading files
        train_file = pd.read_csv(a.train_name)
        test_file = pd.read_csv(a.test_name)

        model = pickle.load(open("/results" + a.model_name + "_" + str(a.add_example) +
                            ".pkl", "rb"))

        predictions = model.predict(test_file)
        return predictions.to_list(), 200





if __name__ == '__main__':
    app.run(debug=True)
