from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.model.networks import *
from src.utils.misc import to_torch, to_numpy, to_cuda, to_cpu
class BaseModel(object):
	"""docstring for BaseModel"""
	def __init__(self, cfg):
		super(BaseModel, self).__init__()
		self.cfg = cfg
		self.name = cfg.NAME
		self.define_network()
		self.define_optimizer_and_scheduler()

	def define_network(self):
		print("Setting up network")	
		self.network = eval(self.cfg.NETWORK.NAME)(**self.cfg.NETWORK)
		self.network = torch.nn.DataParallel(self.network, device_ids=self.cfg.GPUS).cuda()

		if self.cfg.NETWORK.PRETRAINED_WEIGHT_PATH:
			print("Loading Pretrained Weight")
			weight = torch.load(self.cfg.NETWORK.PRETRAINED_WEIGHT_PATH)
			self.network.load_state_dict(weight, strict = False)

	def define_optimizer_and_scheduler(self):
		print("Setting up optimizer and optimizer scheduler")
		self.optimizer = eval('torch.optim.' + self.cfg.OPTIMIZER.NAME)(self.network.parameters(),**self.cfg.OPTIMIZER.PARAMETERS)
		self.scheduler = eval('torch.optim.lr_scheduler.' + self.cfg.OPTIMIZER_SCHEDULE.NAME)(self.optimizer, **self.cfg.OPTIMIZER_SCHEDULE.PARAMETERS)
		self.scheduler.step()

	def criterion(self):
		if self.cfg.has_key('criterion'):
			return eval('loss.' + self.cfg.CRITERION)(model.batch, model.output)

		raise NotImplementedError

	def train(self):
		self.network.train()

	def eval(self):
		self.network.eval()

	def set_batch(self, batch):
		self.batch = batch

	def forward(self):
		self.outputs = self.network(to_cuda(self.batch['input']))
		self.loss 	 = self.criterion()
		self.outputs = to_cpu(self.outputs)

	def step(self):
		self.forward()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

	def update_learning_rate(self):
		self.scheduler.step()

	# set requies_grad=Fasle to avoid computation
	def set_requires_grad(self, nets, requires_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def backward(self):
		raise NotImplementedError

	def define_evaluation(self):
		raise NotImplementedError

	def define_scheduler(self):
		raise NotImplementedError
		
	def eval_result(self):
		raise NotImplementedError