import torch.optim

def get_opt(cfg, model):
	if 'tp' == cfg.type:
		params = get_tp_param2lr(cfg, model)
	else:
		params = model.parameters()
	assert cfg.name in torch.optim.__dict__.keys()
	opt_gen = torch.optim.__dict__[cfg.name]

	opt = opt_gen(params=params, **cfg.kwargs)
	return opt

def get_tp_param2lr(cfg, model):
	assert "lr" in cfg.kwargs.keys()
	blr = cfg.get('blr', cfg.kwargs.lr)
	flr = cfg.get('flr', cfg.kwargs.lr)
	tlr = cfg.get('tlr', cfg.kwargs.lr)
	param_b = [param for param in model.bnetwork.parameters()]
	param_f = []
	for i in range(model.num_layers - 1):
		for param in model.get_params_forward(i + 1):
			param_f.append(param)
	param_t = [param for param in model.get_params_forward(model.num_layers)]
	params = [
			{
				"params":  param_b,
				"lr": blr,
			},
			{
				"params": param_f,
				"lr": flr,
			},
			{
				"params": param_t,
				"lr": tlr,
			},
		]
	return params
