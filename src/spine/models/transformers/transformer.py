from sklearn.preprocessing import FunctionTransformer

def ohe_min_max_transformation(data, data_interface):
    """the data is one-hot-encoded and min-max normalized and fed to the ML model"""
    return data_interface.get_ohe_min_max_normalized_data(data)


def inverse_ohe_min_max_transformation(data, data_interface):
    return data_interface.get_inverse_ohe_min_max_normalized_data(data)

class DataTransfomer:
    """A class to transform data based on user-defined function to get predicted outcomes.
       This class calls FunctionTransformer of scikit-learn internally and is copied from dice_ml.utils
       (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)."""

    def __init__(self, func=None, kw_args=None):
        self.func = func
        self.kw_args = kw_args

    def feed_data_params(self, data_interface):
        if self.kw_args is not None:
            self.kw_args['data_interface'] = data_interface
        else:
            self.kw_args = {'data_interface': data_interface}

    def initialize_transform_func(self):
        if self.func == 'ohe-min-max':
            self.data_transformer = FunctionTransformer(
                    func=ohe_min_max_transformation,
                    inverse_func=inverse_ohe_min_max_transformation,
                    check_inverse=False,
                    validate=False,
                    kw_args=self.kw_args,
                    inv_kw_args=self.kw_args)
        elif self.func is None:
            self.data_transformer = FunctionTransformer(func=self.func, kw_args=None, validate=False)
        else:
            self.data_transformer = FunctionTransformer(func=self.func, kw_args=self.kw_args, validate=False)

    def transform(self, data):
        return self.data_transformer.transform(data)  # should return a numpy array

    def inverse_transform(self, data):
        return self.data_transformer.inverse_transform(data)  # should return a numpy array