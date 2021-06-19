from LightingModel import LitModel

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

model = LitModel()
trainer = pl.Trainer()
trainer.fit(model, DataLoader(train), DataLoader(val))