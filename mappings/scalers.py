from common.misc import create_repository, option
from components.scalers import scaler_pipeline, standard_scaler, minmax_scaler

repository = create_repository({
    'standard_scaler': option(
        standard_scaler.scaler,
        standard_scaler.tests
    ),
    'minmax_scaler': option(
        minmax_scaler.scaler,
        minmax_scaler.tests
    ),
    'scaler_pipeline': option(
        scaler_pipeline.custom_scaler,
        scaler_pipeline.tests
    ),
}, label='scaler')