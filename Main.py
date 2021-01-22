import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# import data
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

real.head()