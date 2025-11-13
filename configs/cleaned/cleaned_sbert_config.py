# Configuration for cleaned data analysis with SBERT
# This config inherits all settings from base_constants and overrides specific ones

from configs.base_constants import *

# Override specific parameters
data_choice = "cleaned"
learning_rate = 0.001  # Lower learning rate for real data
batch_size = 16  # Smaller batch size for memory efficiency
