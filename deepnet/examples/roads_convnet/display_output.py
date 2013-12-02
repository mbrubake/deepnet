from deepnet import deepnet_pb2,neuralnet
import matplotlib.pyplot as plt
import glob, sys, gzip, numpy as np

if __name__ == '__main__':
  proto = sys.argv[1]
  proto = glob.glob(proto + "*")[-1]
  train_txt = sys.argv[2]
  eval_txt = sys.argv[3]

  model_pb = deepnet_pb2.Model()
  f = gzip.open(proto, 'rb')
  model_pb.ParseFromString(f.read())
  f.close()

  nn = neuralnet.NeuralNet(model_pb,train_txt,eval_txt)

  nn.PrintNetwork()

  nn.Evaluate(validation=False,collect_predictions=True)

  raw_input('Press any key')
