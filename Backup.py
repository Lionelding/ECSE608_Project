caffe.set_mode_gpu()


weight=''
prototxt=''
net = caffe.Net(prototxt, caffe.TRAIN)
net = caffe.Net(prototxt, weight, caffe.TEST)


data = net.blobs['data'].data
net.blobs['data'].data[...] = my_image
fc7_activations = net.blobs['fc7'].data


nice_edge_detectors = net.params['conv1'].data
higher_level_filter = net.params['fc7'].data


net.blobs['data'].data[...] = my_image
net.forward() # equivalent to net.forward_all()
softmax_probabilities = net.blobs['prob'].data


net.save('/path/to/new/caffemodel/file')

#instantiate the solver with
solver = caffe.SGDSolver('/path/to/solver/prototxt/file')

training_net = solver.net
 test_net = solver.test_nets[0] # more than one test net is supported
 
 solver.step(1)
 
 solver.solve()
 
Ref:
http://stackoverflow.com/questions/32379878/cheat-sheet-for-caffe-pycaffe?noredirect=1&lq=1
 
 