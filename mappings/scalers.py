from common.interpreter import create_repository, option
from components.scalers import minmax_scaler, standard_scaler

repository = create_repository({
    'standard_scaler': option(
        standard_scaler.custom_scaler,
        standard_scaler.tests
    ),
    'minmax_scaler': option(
        minmax_scaler.custom_scaler,
        minmax_scaler.tests
    ),
}, label='scaler')