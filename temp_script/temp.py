from EBMs_numpy import boltzmann_net_no_hidden
import importlib
importlib.reload(boltzmann_net_no_hidden)

model = boltzmann_net_no_hidden.BoltzmannNetNoHidden(10)
model.b = 0
print(model.b)

model.save_parameters(fileext='pkl', filepathname='./model_save/haha.h5')
model.save_parameters(fileext='.pkl')

model.load_parameters(fileext='.h5')
print(model.b)





