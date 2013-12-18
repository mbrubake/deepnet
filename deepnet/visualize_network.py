"""Push the data through a network and get representations at each layer."""
from neuralnet import *
from trainer import *
import sys
import matplotlib.pyplot as plt

def VisualizeNetwork(model_file, train_op_file):
  if isinstance(model_file, str):
    model = util.ReadModel(model_file)
  else:
    model = model_file
  if isinstance(train_op_file, str):
    op = ReadOperation(train_op_file)
  else:
    op = train_op_file
  op.randomize = False
  op.get_last_piece = True
  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData(skip_outputs=False)
  net.GetTrainBatch()
  net.ForwardPropagate()
  net.Show()

def Usage():
  print 'python %s <model_file> <train_op_file>' % sys.argv[0]

if __name__ == '__main__':
  if len(sys.argv) < 2:
    Usage()
    sys.exit(0)
  board = LockGPU()
  model_file = sys.argv[1]
  model = util.ReadModel(model_file)
  train_op_file = sys.argv[2]
  VisualizeNetwork(model_file, train_op_file)
  FreeGPU(board)
  plt.show(block=True)

