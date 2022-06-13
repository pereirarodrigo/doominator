""" File processing utilities.

The functions in this file are auxiliary in nature, and meant for 
preprocessing game frames and launching ViZDoom game instances for 
training and testing. 

This code was adapted from ViZDoom's PyTorch training example, available 
here: https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_pytorch.py

@author: Rodrigo Pereira
"""
import vizdoom
import numpy as np
import skimage.transform as skt
 

def preprocess(img):
    """
    Preprocesses the images obtained directly from the game to a format
    better suited for the agent.
    """
    img = skt.resize(img, (30, 45))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    
    return img


def create_game(config_path):
    """
    Creates a new instance of a Doom game.
    """
    game = vizdoom.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(False)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    
    print("Doom initialized.")

    return game

