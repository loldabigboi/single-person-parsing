from src.models.one_channel import OneChannelOutputModel
from tensorflow import keras
from src.util.data_loader import DataLoader
from src.util.util import get_paths

model = OneChannelOutputModel()
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])

input_paths, target_paths = get_paths('res\\train\\img', 'res\\train\\seg')
model.fit(input_paths, target_paths, batch_size=64, epochs=5)