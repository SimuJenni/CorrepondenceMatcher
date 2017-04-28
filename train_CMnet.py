from CMNet import CMNet
from datasets.ImageNet import ImageNet
from CMNetTrainer import CMNetTrainer
from Preprocessor import Preprocessor


model = CMNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[64, 64, 3])
trainer = CMNetTrainer(model=model, num_roi=1, dataset=data, pre_processor=preprocessor, num_epochs=10, tag='first_attempt',
                       lr_policy='const', optimizer='adam')
trainer.train()
