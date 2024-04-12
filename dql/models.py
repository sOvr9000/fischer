
from keras.models import Model, clone_model



def create_target_model(model: Model) -> Model:
    '''
    Duplicate the model architecture and copy over weights.
    '''
    if not model._is_compiled:
        raise Exception(f'Please compile the training model before running the simulation.')
    target_model = clone_model(model)
    target_model.set_weights(model.get_weights())
    target_model.compile(getattr(model, 'optimizer', 'adam'), getattr(model, 'loss', 'mse'))
    return target_model


