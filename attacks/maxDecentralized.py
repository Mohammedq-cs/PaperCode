import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def batchMaxDec(inputsTensor, postr):
  tensor_size = inputsTensor.size()
  binary_values = torch.rand(tensor_size)  # Generate binary values with 0.5 probability
  random_tensor = torch.where(binary_values <= 0.5, -postr, postr).to(device)
  res = torch.clamp((inputsTensor + random_tensor), 0, 1).to(device)
  return res
