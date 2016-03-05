from sunnybrook.models.bdrx3DCNN import get_model as bottom_model

def get_model(input_var, target_var, multiply_var):

	test_prediction_bottom, prediction_bottom, loss_bottom, params_bottom = \
		bottom_model(input_var, target_var, multiply_var)

	return test_prediction_bottom, prediction_bottom, loss_bottom, params_bottom