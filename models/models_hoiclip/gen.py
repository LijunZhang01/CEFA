import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
import math

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)
vcoco_hoi_text_label = {(0, 41): 'a photo of a person holding a cup',
                        (16, 80): 'a photo of a person cutting with something',
                        (17, 53): 'a photo of a person cutting a pizza',
                        (0, 53): 'a photo of a person holding a pizza', (2, 80): 'a photo of a person sitting',
                        (8, 53): 'a photo of a person eating a pizza',
                        (9, 80): 'a photo of a person eating with something',
                        (23, 80): 'a photo of a person smiling', (21, 37): 'a photo of a person surfing a surfboard',
                        (0, 73): 'a photo of a person holding a book',
                        (2, 13): 'a photo of a person sitting a bench',
                        (5, 73): 'a photo of a person looking at a book',
                        (27, 73): 'a photo of a person reading a book', (1, 80): 'a photo of a person standing',
                        (22, 36): 'a photo of a person skateboarding a skateboard',
                        (20, 30): 'a photo of a person skiing a skis', (0, 80): 'a photo of a person holding',
                        (8, 80): 'a photo of a person eating', (2, 56): 'a photo of a person sitting a chair',
                        (5, 63): 'a photo of a person looking at a laptop',
                        (19, 63): 'a photo of a person working on computer a laptop',
                        (0, 40): 'a photo of a person holding a wine glass',
                        (24, 40): 'a photo of a person drinking a wine glass',
                        (5, 31): 'a photo of a person looking at a snowboard',
                        (28, 31): 'a photo of a person snowboarding a snowboard',
                        (0, 76): 'a photo of a person holding a scissors',
                        (5, 80): 'a photo of a person looking at something',
                        (5, 76): 'a photo of a person looking at a scissors',
                        (16, 76): 'a photo of a person cutting with a scissors',
                        (17, 80): 'a photo of a person cutting',
                        (5, 37): 'a photo of a person looking at a surfboard',
                        (2, 17): 'a photo of a person sitting a horse',
                        (3, 17): 'a photo of a person riding a horse', (4, 80): 'a photo of a person walking',
                        (5, 29): 'a photo of a person looking at a frisbee', (10, 80): 'a photo of a person jumping',
                        (14, 29): 'a photo of a person throwing a frisbee', (18, 80): 'a photo of a person running',
                        (5, 53): 'a photo of a person looking at a pizza',
                        (0, 48): 'a photo of a person holding a sandwich',
                        (8, 48): 'a photo of a person eating a sandwich',
                        (0, 67): 'a photo of a person holding a cell phone',
                        (19, 80): 'a photo of a person working on computer',
                        (0, 24): 'a photo of a person holding a backpack',
                        (13, 24): 'a photo of a person carrying a backpack', (11, 80): 'a photo of a person laying',
                        (11, 57): 'a photo of a person laying a couch',
                        (0, 17): 'a photo of a person holding a horse', (0, 15): 'a photo of a person holding a cat',
                        (11, 59): 'a photo of a person laying a bed',
                        (15, 29): 'a photo of a person catching a frisbee', (3, 80): 'a photo of a person riding',
                        (12, 67): 'a photo of a person talking on phone a cell phone',
                        (0, 31): 'a photo of a person holding a snowboard',
                        (10, 31): 'a photo of a person jumping a snowboard',
                        (5, 36): 'a photo of a person looking at a skateboard',
                        (10, 36): 'a photo of a person jumping a skateboard',
                        (0, 79): 'a photo of a person holding a toothbrush', (27, 80): 'a photo of a person reading',
                        (0, 39): 'a photo of a person holding a bottle',
                        (24, 39): 'a photo of a person drinking a bottle',
                        (2, 59): 'a photo of a person sitting a bed',
                        (5, 48): 'a photo of a person looking at a sandwich',
                        (0, 30): 'a photo of a person holding a skis',
                        (0, 38): 'a photo of a person holding a tennis racket',
                        (5, 32): 'a photo of a person looking at a sports ball',
                        (6, 38): 'a photo of a person hitting with a tennis racket',
                        (7, 32): 'a photo of a person hitting a sports ball',
                        (5, 0): 'a photo of a person looking at a person',
                        (5, 17): 'a photo of a person looking at a horse',
                        (0, 47): 'a photo of a person holding an apple',
                        (5, 18): 'a photo of a person looking at a sheep',
                        (8, 47): 'a photo of a person eating an apple',
                        (25, 32): 'a photo of a person kicking a sports ball',
                        (0, 44): 'a photo of a person holding a spoon',
                        (5, 55): 'a photo of a person looking at a cake',
                        (8, 55): 'a photo of a person eating a cake',
                        (9, 44): 'a photo of a person eating with a spoon',
                        (0, 63): 'a photo of a person holding a laptop',
                        (6, 80): 'a photo of a person hitting with something',
                        (2, 3): 'a photo of a person sitting a motorcycle',
                        (3, 3): 'a photo of a person riding a motorcycle',
                        (0, 43): 'a photo of a person holding a knife',
                        (5, 43): 'a photo of a person looking at a knife',
                        (16, 43): 'a photo of a person cutting with a knife',
                        (17, 55): 'a photo of a person cutting a cake', (7, 80): 'a photo of a person hitting',
                        (0, 34): 'a photo of a person holding a baseball bat',
                        (6, 34): 'a photo of a person hitting with a baseball bat',
                        (15, 80): 'a photo of a person catching', (2, 57): 'a photo of a person sitting a couch',
                        (0, 77): 'a photo of a person holding a teddy bear',
                        (13, 49): 'a photo of a person carrying an orange',
                        (0, 42): 'a photo of a person holding a fork',
                        (9, 42): 'a photo of a person eating with a fork',
                        (5, 62): 'a photo of a person looking at a tv',
                        (0, 28): 'a photo of a person holding a suitcase',
                        (13, 28): 'a photo of a person carrying a suitcase',
                        (2, 20): 'a photo of a person sitting an elephant',
                        (3, 20): 'a photo of a person riding an elephant',
                        (5, 15): 'a photo of a person looking at a cat',
                        (0, 56): 'a photo of a person holding a chair',
                        (5, 60): 'a photo of a person looking at a dining table',
                        (24, 41): 'a photo of a person drinking a cup', (14, 80): 'a photo of a person throwing',
                        (13, 26): 'a photo of a person carrying a handbag',
                        (5, 16): 'a photo of a person looking at a dog',
                        (0, 46): 'a photo of a person holding a banana',
                        (13, 46): 'a photo of a person carrying a banana',
                        (5, 28): 'a photo of a person looking at a suitcase',
                        (9, 43): 'a photo of a person eating with a knife',
                        (0, 37): 'a photo of a person holding a surfboard',
                        (13, 37): 'a photo of a person carrying a surfboard',
                        (8, 54): 'a photo of a person eating a donut',
                        (0, 0): 'a photo of a person holding a person',
                        (0, 35): 'a photo of a person holding a baseball glove',
                        (0, 65): 'a photo of a person holding a remote',
                        (0, 54): 'a photo of a person holding a donut',
                        (0, 26): 'a photo of a person holding a handbag', (13, 80): 'a photo of a person carrying',
                        (13, 0): 'a photo of a person carrying a person',
                        (0, 32): 'a photo of a person holding a sports ball',
                        (14, 32): 'a photo of a person throwing a sports ball',
                        (5, 54): 'a photo of a person looking at a donut',
                        (0, 1): 'a photo of a person holding a bicycle',
                        (2, 1): 'a photo of a person sitting a bicycle',
                        (3, 1): 'a photo of a person riding a bicycle',
                        (5, 1): 'a photo of a person looking at a bicycle', (25, 80): 'a photo of a person kicking',
                        (5, 67): 'a photo of a person looking at a cell phone',
                        (5, 6): 'a photo of a person looking at a train',
                        (0, 29): 'a photo of a person holding a frisbee',
                        (0, 36): 'a photo of a person holding a skateboard',
                        (3, 7): 'a photo of a person riding a truck',
                        (26, 63): 'a photo of a person pointing a laptop',
                        (0, 3): 'a photo of a person holding a motorcycle',
                        (13, 30): 'a photo of a person carrying a skis',
                        (0, 25): 'a photo of a person holding a umbrella',
                        (5, 45): 'a photo of a person looking at a bowl',
                        (17, 51): 'a photo of a person cutting a carrot',
                        (0, 52): 'a photo of a person holding a hot dog',
                        (8, 52): 'a photo of a person eating a hot dog',
                        (0, 33): 'a photo of a person holding a kite',
                        (5, 13): 'a photo of a person looking at a bench',
                        (12, 80): 'a photo of a person talking on phone',
                        (22, 80): 'a photo of a person skateboarding',
                        (5, 35): 'a photo of a person looking at a baseball glove',
                        (15, 32): 'a photo of a person catching a sports ball',
                        (26, 80): 'a photo of a person pointing',
                        (13, 25): 'a photo of a person carrying a umbrella',
                        (5, 40): 'a photo of a person looking at a wine glass',
                        (10, 37): 'a photo of a person jumping a surfboard',
                        (5, 33): 'a photo of a person looking at a kite',
                        (13, 33): 'a photo of a person carrying a kite',
                        (3, 6): 'a photo of a person riding a train',
                        (5, 44): 'a photo of a person looking at a spoon',
                        (0, 20): 'a photo of a person holding an elephant', (21, 80): 'a photo of a person surfing',
                        (5, 20): 'a photo of a person looking at an elephant',
                        (3, 8): 'a photo of a person riding a boat',
                        (5, 23): 'a photo of a person looking at a giraffe',
                        (13, 67): 'a photo of a person carrying a cell phone',
                        (11, 56): 'a photo of a person laying a chair',
                        (5, 19): 'a photo of a person looking at a cow',
                        (5, 42): 'a photo of a person looking at a fork',
                        (0, 55): 'a photo of a person holding a cake',
                        (13, 32): 'a photo of a person carrying a sports ball',
                        (5, 30): 'a photo of a person looking at a skis',
                        (13, 36): 'a photo of a person carrying a skateboard',
                        (26, 67): 'a photo of a person pointing a cell phone',
                        (5, 52): 'a photo of a person looking at a hot dog',
                        (8, 46): 'a photo of a person eating a banana', (20, 80): 'a photo of a person skiing',
                        (28, 80): 'a photo of a person snowboarding', (0, 14): 'a photo of a person holding a bird',
                        (11, 60): 'a photo of a person laying a dining table',
                        (0, 16): 'a photo of a person holding a dog',
                        (0, 72): 'a photo of a person holding a refrigerator',
                        (5, 72): 'a photo of a person looking at a refrigerator',
                        (5, 7): 'a photo of a person looking at a truck',
                        (5, 41): 'a photo of a person looking at a cup',
                        (2, 61): 'a photo of a person sitting a toilet', (24, 80): 'a photo of a person drinking',
                        (0, 27): 'a photo of a person holding a tie',
                        (5, 27): 'a photo of a person looking at a tie',
                        (17, 27): 'a photo of a person cutting a tie',
                        (5, 10): 'a photo of a person looking at a fire hydrant',
                        (26, 10): 'a photo of a person pointing a fire hydrant',
                        (11, 13): 'a photo of a person laying a bench',
                        (17, 18): 'a photo of a person cutting a sheep',
                        (0, 64): 'a photo of a person holding a mouse',
                        (5, 64): 'a photo of a person looking at a mouse',
                        (5, 66): 'a photo of a person looking at a keyboard',
                        (16, 42): 'a photo of a person cutting with a fork',
                        (17, 0): 'a photo of a person cutting a person',
                        (5, 5): 'a photo of a person looking at a bus', (3, 2): 'a photo of a person riding a car',
                        (10, 30): 'a photo of a person jumping a skis',
                        (5, 4): 'a photo of a person looking at an airplane',
                        (5, 46): 'a photo of a person looking at a banana',
                        (2, 28): 'a photo of a person sitting a suitcase',
                        (13, 29): 'a photo of a person carrying a frisbee',
                        (5, 26): 'a photo of a person looking at a handbag',
                        (8, 50): 'a photo of a person eating a broccoli',
                        (17, 46): 'a photo of a person cutting a banana',
                        (0, 18): 'a photo of a person holding a sheep',
                        (17, 48): 'a photo of a person cutting a sandwich',
                        (26, 0): 'a photo of a person pointing a person',
                        (5, 3): 'a photo of a person looking at a motorcycle',
                        (5, 24): 'a photo of a person looking at a backpack',
                        (0, 45): 'a photo of a person holding a bowl',
                        (26, 27): 'a photo of a person pointing a tie',
                        (0, 49): 'a photo of a person holding an orange',
                        (8, 49): 'a photo of a person eating an orange',
                        (5, 34): 'a photo of a person looking at a baseball bat',
                        (13, 31): 'a photo of a person carrying a snowboard',
                        (17, 54): 'a photo of a person cutting a donut',
                        (5, 38): 'a photo of a person looking at a tennis racket',
                        (8, 51): 'a photo of a person eating a carrot',
                        (17, 47): 'a photo of a person cutting an apple',
                        (13, 40): 'a photo of a person carrying a wine glass',
                        (26, 48): 'a photo of a person pointing a sandwich',
                        (26, 62): 'a photo of a person pointing a tv',
                        (13, 74): 'a photo of a person carrying a clock',
                        (5, 61): 'a photo of a person looking at a toilet',
                        (26, 19): 'a photo of a person pointing a cow',
                        (5, 65): 'a photo of a person looking at a remote',
                        (26, 18): 'a photo of a person pointing a sheep',
                        (0, 50): 'a photo of a person holding a broccoli',
                        (0, 13): 'a photo of a person holding a bench',
                        (26, 33): 'a photo of a person pointing a kite',
                        (0, 7): 'a photo of a person holding a truck',
                        (13, 41): 'a photo of a person carrying a cup',
                        (24, 45): 'a photo of a person drinking a bowl',
                        (13, 38): 'a photo of a person carrying a tennis racket',
                        (13, 39): 'a photo of a person carrying a bottle',
                        (5, 47): 'a photo of a person looking at an apple',
                        (5, 56): 'a photo of a person looking at a chair',
                        (2, 24): 'a photo of a person sitting a backpack',
                        (26, 60): 'a photo of a person pointing a dining table',
                        (0, 78): 'a photo of a person holding a hair drier',
                        (5, 39): 'a photo of a person looking at a bottle',
                        (26, 55): 'a photo of a person pointing a cake',
                        (26, 66): 'a photo of a person pointing a keyboard',
                        (26, 72): 'a photo of a person pointing a refrigerator',
                        (5, 74): 'a photo of a person looking at a clock',
                        (0, 8): 'a photo of a person holding a boat', (17, 45): 'a photo of a person cutting a bowl',
                        (26, 23): 'a photo of a person pointing a giraffe',
                        (5, 25): 'a photo of a person looking at a umbrella',
                        (0, 66): 'a photo of a person holding a keyboard',
                        (2, 26): 'a photo of a person sitting a handbag',
                        (26, 52): 'a photo of a person pointing a hot dog',
                        (2, 60): 'a photo of a person sitting a dining table',
                        (13, 77): 'a photo of a person carrying a teddy bear',
                        (0, 51): 'a photo of a person holding a carrot',
                        (13, 34): 'a photo of a person carrying a baseball bat',
                        (5, 2): 'a photo of a person looking at a car', (3, 5): 'a photo of a person riding a bus',
                        (17, 50): 'a photo of a person cutting a broccoli',
                        (5, 14): 'a photo of a person looking at a bird',
                        (13, 73): 'a photo of a person carrying a book',
                        (5, 50): 'a photo of a person looking at a broccoli'}

hico_text_label = {(4, 4): 'a photo of a person boarding an airplane',
                   (17, 4): 'a photo of a person directing an airplane',
                   (25, 4): 'a photo of a person exiting an airplane',
                   (30, 4): 'a photo of a person flying an airplane',
                   (41, 4): 'a photo of a person inspecting an airplane',
                   (52, 4): 'a photo of a person loading an airplane',
                   (76, 4): 'a photo of a person riding an airplane',
                   (87, 4): 'a photo of a person sitting on an airplane',
                   (111, 4): 'a photo of a person washing an airplane',
                   (57, 4): 'a photo of a person and an airplane', (8, 1): 'a photo of a person carrying a bicycle',
                   (36, 1): 'a photo of a person holding a bicycle',
                   (41, 1): 'a photo of a person inspecting a bicycle',
                   (43, 1): 'a photo of a person jumping a bicycle',
                   (37, 1): 'a photo of a person hopping on a bicycle',
                   (62, 1): 'a photo of a person parking a bicycle',
                   (71, 1): 'a photo of a person pushing a bicycle',
                   (75, 1): 'a photo of a person repairing a bicycle',
                   (76, 1): 'a photo of a person riding a bicycle',
                   (87, 1): 'a photo of a person sitting on a bicycle',
                   (98, 1): 'a photo of a person straddling a bicycle',
                   (110, 1): 'a photo of a person walking a bicycle',
                   (111, 1): 'a photo of a person washing a bicycle', (57, 1): 'a photo of a person and a bicycle',
                   (10, 14): 'a photo of a person chasing a bird', (26, 14): 'a photo of a person feeding a bird',
                   (36, 14): 'a photo of a person holding a bird', (65, 14): 'a photo of a person petting a bird',
                   (74, 14): 'a photo of a person releasing a bird',
                   (112, 14): 'a photo of a person watching a bird', (57, 14): 'a photo of a person and a bird',
                   (4, 8): 'a photo of a person boarding a boat', (21, 8): 'a photo of a person driving a boat',
                   (25, 8): 'a photo of a person exiting a boat', (41, 8): 'a photo of a person inspecting a boat',
                   (43, 8): 'a photo of a person jumping a boat', (47, 8): 'a photo of a person launching a boat',
                   (75, 8): 'a photo of a person repairing a boat', (76, 8): 'a photo of a person riding a boat',
                   (77, 8): 'a photo of a person rowing a boat', (79, 8): 'a photo of a person sailing a boat',
                   (87, 8): 'a photo of a person sitting on a boat',
                   (93, 8): 'a photo of a person standing on a boat', (105, 8): 'a photo of a person tying a boat',
                   (111, 8): 'a photo of a person washing a boat', (57, 8): 'a photo of a person and a boat',
                   (8, 39): 'a photo of a person carrying a bottle',
                   (20, 39): 'a photo of a person drinking with a bottle',
                   (36, 39): 'a photo of a person holding a bottle',
                   (41, 39): 'a photo of a person inspecting a bottle',
                   (48, 39): 'a photo of a person licking a bottle',
                   (58, 39): 'a photo of a person opening a bottle',
                   (69, 39): 'a photo of a person pouring a bottle', (57, 39): 'a photo of a person and a bottle',
                   (4, 5): 'a photo of a person boarding a bus', (17, 5): 'a photo of a person directing a bus',
                   (21, 5): 'a photo of a person driving a bus', (25, 5): 'a photo of a person exiting a bus',
                   (41, 5): 'a photo of a person inspecting a bus', (52, 5): 'a photo of a person loading a bus',
                   (76, 5): 'a photo of a person riding a bus', (87, 5): 'a photo of a person sitting on a bus',
                   (111, 5): 'a photo of a person washing a bus', (113, 5): 'a photo of a person waving a bus',
                   (57, 5): 'a photo of a person and a bus', (4, 2): 'a photo of a person boarding a car',
                   (17, 2): 'a photo of a person directing a car', (21, 2): 'a photo of a person driving a car',
                   (38, 2): 'a photo of a person hosing a car', (41, 2): 'a photo of a person inspecting a car',
                   (43, 2): 'a photo of a person jumping a car', (52, 2): 'a photo of a person loading a car',
                   (62, 2): 'a photo of a person parking a car', (76, 2): 'a photo of a person riding a car',
                   (111, 2): 'a photo of a person washing a car', (57, 2): 'a photo of a person and a car',
                   (22, 15): 'a photo of a person drying a cat', (26, 15): 'a photo of a person feeding a cat',
                   (36, 15): 'a photo of a person holding a cat', (39, 15): 'a photo of a person hugging a cat',
                   (45, 15): 'a photo of a person kissing a cat', (65, 15): 'a photo of a person petting a cat',
                   (80, 15): 'a photo of a person scratching a cat', (111, 15): 'a photo of a person washing a cat',
                   (10, 15): 'a photo of a person chasing a cat', (57, 15): 'a photo of a person and a cat',
                   (8, 56): 'a photo of a person carrying a chair', (36, 56): 'a photo of a person holding a chair',
                   (49, 56): 'a photo of a person lying on a chair',
                   (87, 56): 'a photo of a person sitting on a chair',
                   (93, 56): 'a photo of a person standing on a chair', (57, 56): 'a photo of a person and a chair',
                   (8, 57): 'a photo of a person carrying a couch',
                   (49, 57): 'a photo of a person lying on a couch',
                   (87, 57): 'a photo of a person sitting on a couch', (57, 57): 'a photo of a person and a couch',
                   (26, 19): 'a photo of a person feeding a cow', (34, 19): 'a photo of a person herding a cow',
                   (36, 19): 'a photo of a person holding a cow', (39, 19): 'a photo of a person hugging a cow',
                   (45, 19): 'a photo of a person kissing a cow', (46, 19): 'a photo of a person lassoing a cow',
                   (55, 19): 'a photo of a person milking a cow', (65, 19): 'a photo of a person petting a cow',
                   (76, 19): 'a photo of a person riding a cow', (110, 19): 'a photo of a person walking a cow',
                   (57, 19): 'a photo of a person and a cow',
                   (12, 60): 'a photo of a person cleaning a dining table',
                   (24, 60): 'a photo of a person eating at a dining table',
                   (86, 60): 'a photo of a person sitting at a dining table',
                   (57, 60): 'a photo of a person and a dining table',
                   (8, 16): 'a photo of a person carrying a dog', (22, 16): 'a photo of a person drying a dog',
                   (26, 16): 'a photo of a person feeding a dog', (33, 16): 'a photo of a person grooming a dog',
                   (36, 16): 'a photo of a person holding a dog', (38, 16): 'a photo of a person hosing a dog',
                   (39, 16): 'a photo of a person hugging a dog', (41, 16): 'a photo of a person inspecting a dog',
                   (45, 16): 'a photo of a person kissing a dog', (65, 16): 'a photo of a person petting a dog',
                   (78, 16): 'a photo of a person running a dog', (80, 16): 'a photo of a person scratching a dog',
                   (98, 16): 'a photo of a person straddling a dog',
                   (107, 16): 'a photo of a person training a dog', (110, 16): 'a photo of a person walking a dog',
                   (111, 16): 'a photo of a person washing a dog', (10, 16): 'a photo of a person chasing a dog',
                   (57, 16): 'a photo of a person and a dog', (26, 17): 'a photo of a person feeding a horse',
                   (33, 17): 'a photo of a person grooming a horse',
                   (36, 17): 'a photo of a person holding a horse', (39, 17): 'a photo of a person hugging a horse',
                   (43, 17): 'a photo of a person jumping a horse', (45, 17): 'a photo of a person kissing a horse',
                   (52, 17): 'a photo of a person loading a horse',
                   (37, 17): 'a photo of a person hopping on a horse',
                   (65, 17): 'a photo of a person petting a horse', (72, 17): 'a photo of a person racing a horse',
                   (76, 17): 'a photo of a person riding a horse', (78, 17): 'a photo of a person running a horse',
                   (98, 17): 'a photo of a person straddling a horse',
                   (107, 17): 'a photo of a person training a horse',
                   (110, 17): 'a photo of a person walking a horse',
                   (111, 17): 'a photo of a person washing a horse', (57, 17): 'a photo of a person and a horse',
                   (36, 3): 'a photo of a person holding a motorcycle',
                   (41, 3): 'a photo of a person inspecting a motorcycle',
                   (43, 3): 'a photo of a person jumping a motorcycle',
                   (37, 3): 'a photo of a person hopping on a motorcycle',
                   (62, 3): 'a photo of a person parking a motorcycle',
                   (71, 3): 'a photo of a person pushing a motorcycle',
                   (72, 3): 'a photo of a person racing a motorcycle',
                   (76, 3): 'a photo of a person riding a motorcycle',
                   (87, 3): 'a photo of a person sitting on a motorcycle',
                   (98, 3): 'a photo of a person straddling a motorcycle',
                   (108, 3): 'a photo of a person turning a motorcycle',
                   (110, 3): 'a photo of a person walking a motorcycle',
                   (111, 3): 'a photo of a person washing a motorcycle',
                   (57, 3): 'a photo of a person and a motorcycle', (8, 0): 'a photo of a person carrying a person',
                   (31, 0): 'a photo of a person greeting a person',
                   (36, 0): 'a photo of a person holding a person', (39, 0): 'a photo of a person hugging a person',
                   (45, 0): 'a photo of a person kissing a person',
                   (92, 0): 'a photo of a person stabbing a person',
                   (100, 0): 'a photo of a person tagging a person',
                   (102, 0): 'a photo of a person teaching a person',
                   (48, 0): 'a photo of a person licking a person', (57, 0): 'a photo of a person and a person',
                   (8, 58): 'a photo of a person carrying a potted plant',
                   (36, 58): 'a photo of a person holding a potted plant',
                   (38, 58): 'a photo of a person hosing a potted plant',
                   (57, 58): 'a photo of a person and a potted plant',
                   (8, 18): 'a photo of a person carrying a sheep', (26, 18): 'a photo of a person feeding a sheep',
                   (34, 18): 'a photo of a person herding a sheep', (36, 18): 'a photo of a person holding a sheep',
                   (39, 18): 'a photo of a person hugging a sheep', (45, 18): 'a photo of a person kissing a sheep',
                   (65, 18): 'a photo of a person petting a sheep', (76, 18): 'a photo of a person riding a sheep',
                   (83, 18): 'a photo of a person shearing a sheep',
                   (110, 18): 'a photo of a person walking a sheep',
                   (111, 18): 'a photo of a person washing a sheep', (57, 18): 'a photo of a person and a sheep',
                   (4, 6): 'a photo of a person boarding a train', (21, 6): 'a photo of a person driving a train',
                   (25, 6): 'a photo of a person exiting a train', (52, 6): 'a photo of a person loading a train',
                   (76, 6): 'a photo of a person riding a train', (87, 6): 'a photo of a person sitting on a train',
                   (111, 6): 'a photo of a person washing a train', (57, 6): 'a photo of a person and a train',
                   (13, 62): 'a photo of a person controlling a tv', (75, 62): 'a photo of a person repairing a tv',
                   (112, 62): 'a photo of a person watching a tv', (57, 62): 'a photo of a person and a tv',
                   (7, 47): 'a photo of a person buying an apple', (15, 47): 'a photo of a person cutting an apple',
                   (23, 47): 'a photo of a person eating an apple',
                   (36, 47): 'a photo of a person holding an apple',
                   (41, 47): 'a photo of a person inspecting an apple',
                   (64, 47): 'a photo of a person peeling an apple',
                   (66, 47): 'a photo of a person picking an apple',
                   (89, 47): 'a photo of a person smelling an apple',
                   (111, 47): 'a photo of a person washing an apple', (57, 47): 'a photo of a person and an apple',
                   (8, 24): 'a photo of a person carrying a backpack',
                   (36, 24): 'a photo of a person holding a backpack',
                   (41, 24): 'a photo of a person inspecting a backpack',
                   (58, 24): 'a photo of a person opening a backpack',
                   (114, 24): 'a photo of a person wearing a backpack',
                   (57, 24): 'a photo of a person and a backpack', (7, 46): 'a photo of a person buying a banana',
                   (8, 46): 'a photo of a person carrying a banana',
                   (15, 46): 'a photo of a person cutting a banana',
                   (23, 46): 'a photo of a person eating a banana',
                   (36, 46): 'a photo of a person holding a banana',
                   (41, 46): 'a photo of a person inspecting a banana',
                   (64, 46): 'a photo of a person peeling a banana',
                   (66, 46): 'a photo of a person picking a banana',
                   (89, 46): 'a photo of a person smelling a banana', (57, 46): 'a photo of a person and a banana',
                   (5, 34): 'a photo of a person breaking a baseball bat',
                   (8, 34): 'a photo of a person carrying a baseball bat',
                   (36, 34): 'a photo of a person holding a baseball bat',
                   (84, 34): 'a photo of a person signing a baseball bat',
                   (99, 34): 'a photo of a person swinging a baseball bat',
                   (104, 34): 'a photo of a person throwing a baseball bat',
                   (115, 34): 'a photo of a person wielding a baseball bat',
                   (57, 34): 'a photo of a person and a baseball bat',
                   (36, 35): 'a photo of a person holding a baseball glove',
                   (114, 35): 'a photo of a person wearing a baseball glove',
                   (57, 35): 'a photo of a person and a baseball glove',
                   (26, 21): 'a photo of a person feeding a bear', (40, 21): 'a photo of a person hunting a bear',
                   (112, 21): 'a photo of a person watching a bear', (57, 21): 'a photo of a person and a bear',
                   (12, 59): 'a photo of a person cleaning a bed', (49, 59): 'a photo of a person lying on a bed',
                   (87, 59): 'a photo of a person sitting on a bed', (57, 59): 'a photo of a person and a bed',
                   (41, 13): 'a photo of a person inspecting a bench',
                   (49, 13): 'a photo of a person lying on a bench',
                   (87, 13): 'a photo of a person sitting on a bench', (57, 13): 'a photo of a person and a bench',
                   (8, 73): 'a photo of a person carrying a book', (36, 73): 'a photo of a person holding a book',
                   (58, 73): 'a photo of a person opening a book', (73, 73): 'a photo of a person reading a book',
                   (57, 73): 'a photo of a person and a book', (36, 45): 'a photo of a person holding a bowl',
                   (96, 45): 'a photo of a person stirring a bowl', (111, 45): 'a photo of a person washing a bowl',
                   (48, 45): 'a photo of a person licking a bowl', (57, 45): 'a photo of a person and a bowl',
                   (15, 50): 'a photo of a person cutting a broccoli',
                   (23, 50): 'a photo of a person eating a broccoli',
                   (36, 50): 'a photo of a person holding a broccoli',
                   (89, 50): 'a photo of a person smelling a broccoli',
                   (96, 50): 'a photo of a person stirring a broccoli',
                   (111, 50): 'a photo of a person washing a broccoli',
                   (57, 50): 'a photo of a person and a broccoli', (3, 55): 'a photo of a person blowing a cake',
                   (8, 55): 'a photo of a person carrying a cake', (15, 55): 'a photo of a person cutting a cake',
                   (23, 55): 'a photo of a person eating a cake', (36, 55): 'a photo of a person holding a cake',
                   (51, 55): 'a photo of a person lighting a cake', (54, 55): 'a photo of a person making a cake',
                   (67, 55): 'a photo of a person picking up a cake', (57, 55): 'a photo of a person and a cake',
                   (8, 51): 'a photo of a person carrying a carrot',
                   (14, 51): 'a photo of a person cooking a carrot',
                   (15, 51): 'a photo of a person cutting a carrot',
                   (23, 51): 'a photo of a person eating a carrot',
                   (36, 51): 'a photo of a person holding a carrot',
                   (64, 51): 'a photo of a person peeling a carrot',
                   (89, 51): 'a photo of a person smelling a carrot',
                   (96, 51): 'a photo of a person stirring a carrot',
                   (111, 51): 'a photo of a person washing a carrot', (57, 51): 'a photo of a person and a carrot',
                   (8, 67): 'a photo of a person carrying a cell phone',
                   (36, 67): 'a photo of a person holding a cell phone',
                   (73, 67): 'a photo of a person reading a cell phone',
                   (75, 67): 'a photo of a person repairing a cell phone',
                   (101, 67): 'a photo of a person talking on a cell phone',
                   (103, 67): 'a photo of a person texting on a cell phone',
                   (57, 67): 'a photo of a person and a cell phone',
                   (11, 74): 'a photo of a person checking a clock',
                   (36, 74): 'a photo of a person holding a clock',
                   (75, 74): 'a photo of a person repairing a clock',
                   (82, 74): 'a photo of a person setting a clock', (57, 74): 'a photo of a person and a clock',
                   (8, 41): 'a photo of a person carrying a cup',
                   (20, 41): 'a photo of a person drinking with a cup',
                   (36, 41): 'a photo of a person holding a cup', (41, 41): 'a photo of a person inspecting a cup',
                   (69, 41): 'a photo of a person pouring a cup', (85, 41): 'a photo of a person sipping a cup',
                   (89, 41): 'a photo of a person smelling a cup', (27, 41): 'a photo of a person filling a cup',
                   (111, 41): 'a photo of a person washing a cup', (57, 41): 'a photo of a person and a cup',
                   (7, 54): 'a photo of a person buying a donut', (8, 54): 'a photo of a person carrying a donut',
                   (23, 54): 'a photo of a person eating a donut', (36, 54): 'a photo of a person holding a donut',
                   (54, 54): 'a photo of a person making a donut',
                   (67, 54): 'a photo of a person picking up a donut',
                   (89, 54): 'a photo of a person smelling a donut', (57, 54): 'a photo of a person and a donut',
                   (26, 20): 'a photo of a person feeding an elephant',
                   (36, 20): 'a photo of a person holding an elephant',
                   (38, 20): 'a photo of a person hosing an elephant',
                   (39, 20): 'a photo of a person hugging an elephant',
                   (45, 20): 'a photo of a person kissing an elephant',
                   (37, 20): 'a photo of a person hopping on an elephant',
                   (65, 20): 'a photo of a person petting an elephant',
                   (76, 20): 'a photo of a person riding an elephant',
                   (110, 20): 'a photo of a person walking an elephant',
                   (111, 20): 'a photo of a person washing an elephant',
                   (112, 20): 'a photo of a person watching an elephant',
                   (57, 20): 'a photo of a person and an elephant',
                   (39, 10): 'a photo of a person hugging a fire hydrant',
                   (41, 10): 'a photo of a person inspecting a fire hydrant',
                   (58, 10): 'a photo of a person opening a fire hydrant',
                   (61, 10): 'a photo of a person painting a fire hydrant',
                   (57, 10): 'a photo of a person and a fire hydrant',
                   (36, 42): 'a photo of a person holding a fork', (50, 42): 'a photo of a person lifting a fork',
                   (95, 42): 'a photo of a person sticking a fork', (48, 42): 'a photo of a person licking a fork',
                   (111, 42): 'a photo of a person washing a fork', (57, 42): 'a photo of a person and a fork',
                   (2, 29): 'a photo of a person blocking a frisbee',
                   (9, 29): 'a photo of a person catching a frisbee',
                   (36, 29): 'a photo of a person holding a frisbee',
                   (90, 29): 'a photo of a person spinning a frisbee',
                   (104, 29): 'a photo of a person throwing a frisbee',
                   (57, 29): 'a photo of a person and a frisbee', (26, 23): 'a photo of a person feeding a giraffe',
                   (45, 23): 'a photo of a person kissing a giraffe',
                   (65, 23): 'a photo of a person petting a giraffe',
                   (76, 23): 'a photo of a person riding a giraffe',
                   (112, 23): 'a photo of a person watching a giraffe',
                   (57, 23): 'a photo of a person and a giraffe',
                   (36, 78): 'a photo of a person holding a hair drier',
                   (59, 78): 'a photo of a person operating a hair drier',
                   (75, 78): 'a photo of a person repairing a hair drier',
                   (57, 78): 'a photo of a person and a hair drier',
                   (8, 26): 'a photo of a person carrying a handbag',
                   (36, 26): 'a photo of a person holding a handbag',
                   (41, 26): 'a photo of a person inspecting a handbag',
                   (57, 26): 'a photo of a person and a handbag', (8, 52): 'a photo of a person carrying a hot dog',
                   (14, 52): 'a photo of a person cooking a hot dog',
                   (15, 52): 'a photo of a person cutting a hot dog',
                   (23, 52): 'a photo of a person eating a hot dog',
                   (36, 52): 'a photo of a person holding a hot dog',
                   (54, 52): 'a photo of a person making a hot dog', (57, 52): 'a photo of a person and a hot dog',
                   (8, 66): 'a photo of a person carrying a keyboard',
                   (12, 66): 'a photo of a person cleaning a keyboard',
                   (36, 66): 'a photo of a person holding a keyboard',
                   (109, 66): 'a photo of a person typing on a keyboard',
                   (57, 66): 'a photo of a person and a keyboard', (1, 33): 'a photo of a person assembling a kite',
                   (8, 33): 'a photo of a person carrying a kite', (30, 33): 'a photo of a person flying a kite',
                   (36, 33): 'a photo of a person holding a kite',
                   (41, 33): 'a photo of a person inspecting a kite',
                   (47, 33): 'a photo of a person launching a kite', (70, 33): 'a photo of a person pulling a kite',
                   (57, 33): 'a photo of a person and a kite', (16, 43): 'a photo of a person cutting with a knife',
                   (36, 43): 'a photo of a person holding a knife',
                   (95, 43): 'a photo of a person sticking a knife',
                   (111, 43): 'a photo of a person washing a knife',
                   (115, 43): 'a photo of a person wielding a knife',
                   (48, 43): 'a photo of a person licking a knife', (57, 43): 'a photo of a person and a knife',
                   (36, 63): 'a photo of a person holding a laptop',
                   (58, 63): 'a photo of a person opening a laptop',
                   (73, 63): 'a photo of a person reading a laptop',
                   (75, 63): 'a photo of a person repairing a laptop',
                   (109, 63): 'a photo of a person typing on a laptop',
                   (57, 63): 'a photo of a person and a laptop',
                   (12, 68): 'a photo of a person cleaning a microwave',
                   (58, 68): 'a photo of a person opening a microwave',
                   (59, 68): 'a photo of a person operating a microwave',
                   (57, 68): 'a photo of a person and a microwave',
                   (13, 64): 'a photo of a person controlling a mouse',
                   (36, 64): 'a photo of a person holding a mouse',
                   (75, 64): 'a photo of a person repairing a mouse', (57, 64): 'a photo of a person and a mouse',
                   (7, 49): 'a photo of a person buying an orange',
                   (15, 49): 'a photo of a person cutting an orange',
                   (23, 49): 'a photo of a person eating an orange',
                   (36, 49): 'a photo of a person holding an orange',
                   (41, 49): 'a photo of a person inspecting an orange',
                   (64, 49): 'a photo of a person peeling an orange',
                   (66, 49): 'a photo of a person picking an orange',
                   (91, 49): 'a photo of a person squeezing an orange',
                   (111, 49): 'a photo of a person washing an orange',
                   (57, 49): 'a photo of a person and an orange', (12, 69): 'a photo of a person cleaning an oven',
                   (36, 69): 'a photo of a person holding an oven',
                   (41, 69): 'a photo of a person inspecting an oven',
                   (58, 69): 'a photo of a person opening an oven',
                   (75, 69): 'a photo of a person repairing an oven',
                   (59, 69): 'a photo of a person operating an oven', (57, 69): 'a photo of a person and an oven',
                   (11, 12): 'a photo of a person checking a parking meter',
                   (63, 12): 'a photo of a person paying a parking meter',
                   (75, 12): 'a photo of a person repairing a parking meter',
                   (57, 12): 'a photo of a person and a parking meter',
                   (7, 53): 'a photo of a person buying a pizza', (8, 53): 'a photo of a person carrying a pizza',
                   (14, 53): 'a photo of a person cooking a pizza', (15, 53): 'a photo of a person cutting a pizza',
                   (23, 53): 'a photo of a person eating a pizza', (36, 53): 'a photo of a person holding a pizza',
                   (54, 53): 'a photo of a person making a pizza',
                   (67, 53): 'a photo of a person picking up a pizza',
                   (88, 53): 'a photo of a person sliding a pizza',
                   (89, 53): 'a photo of a person smelling a pizza', (57, 53): 'a photo of a person and a pizza',
                   (12, 72): 'a photo of a person cleaning a refrigerator',
                   (36, 72): 'a photo of a person holding a refrigerator',
                   (56, 72): 'a photo of a person moving a refrigerator',
                   (58, 72): 'a photo of a person opening a refrigerator',
                   (57, 72): 'a photo of a person and a refrigerator',
                   (36, 65): 'a photo of a person holding a remote',
                   (68, 65): 'a photo of a person pointing a remote',
                   (99, 65): 'a photo of a person swinging a remote', (57, 65): 'a photo of a person and a remote',
                   (8, 48): 'a photo of a person carrying a sandwich',
                   (14, 48): 'a photo of a person cooking a sandwich',
                   (15, 48): 'a photo of a person cutting a sandwich',
                   (23, 48): 'a photo of a person eating a sandwich',
                   (36, 48): 'a photo of a person holding a sandwich',
                   (54, 48): 'a photo of a person making a sandwich',
                   (57, 48): 'a photo of a person and a sandwich',
                   (16, 76): 'a photo of a person cutting with a scissors',
                   (36, 76): 'a photo of a person holding a scissors',
                   (58, 76): 'a photo of a person opening a scissors',
                   (57, 76): 'a photo of a person and a scissors', (12, 71): 'a photo of a person cleaning a sink',
                   (75, 71): 'a photo of a person repairing a sink',
                   (111, 71): 'a photo of a person washing a sink', (57, 71): 'a photo of a person and a sink',
                   (8, 36): 'a photo of a person carrying a skateboard',
                   (28, 36): 'a photo of a person flipping a skateboard',
                   (32, 36): 'a photo of a person grinding a skateboard',
                   (36, 36): 'a photo of a person holding a skateboard',
                   (43, 36): 'a photo of a person jumping a skateboard',
                   (67, 36): 'a photo of a person picking up a skateboard',
                   (76, 36): 'a photo of a person riding a skateboard',
                   (87, 36): 'a photo of a person sitting on a skateboard',
                   (93, 36): 'a photo of a person standing on a skateboard',
                   (57, 36): 'a photo of a person and a skateboard',
                   (0, 30): 'a photo of a person adjusting a skis', (8, 30): 'a photo of a person carrying a skis',
                   (36, 30): 'a photo of a person holding a skis',
                   (41, 30): 'a photo of a person inspecting a skis',
                   (43, 30): 'a photo of a person jumping a skis',
                   (67, 30): 'a photo of a person picking up a skis',
                   (75, 30): 'a photo of a person repairing a skis', (76, 30): 'a photo of a person riding a skis',
                   (93, 30): 'a photo of a person standing on a skis',
                   (114, 30): 'a photo of a person wearing a skis', (57, 30): 'a photo of a person and a skis',
                   (0, 31): 'a photo of a person adjusting a snowboard',
                   (8, 31): 'a photo of a person carrying a snowboard',
                   (32, 31): 'a photo of a person grinding a snowboard',
                   (36, 31): 'a photo of a person holding a snowboard',
                   (43, 31): 'a photo of a person jumping a snowboard',
                   (76, 31): 'a photo of a person riding a snowboard',
                   (93, 31): 'a photo of a person standing on a snowboard',
                   (114, 31): 'a photo of a person wearing a snowboard',
                   (57, 31): 'a photo of a person and a snowboard', (36, 44): 'a photo of a person holding a spoon',
                   (48, 44): 'a photo of a person licking a spoon',
                   (111, 44): 'a photo of a person washing a spoon',
                   (85, 44): 'a photo of a person sipping a spoon', (57, 44): 'a photo of a person and a spoon',
                   (2, 32): 'a photo of a person blocking a sports ball',
                   (8, 32): 'a photo of a person carrying a sports ball',
                   (9, 32): 'a photo of a person catching a sports ball',
                   (19, 32): 'a photo of a person dribbling a sports ball',
                   (35, 32): 'a photo of a person hitting a sports ball',
                   (36, 32): 'a photo of a person holding a sports ball',
                   (41, 32): 'a photo of a person inspecting a sports ball',
                   (44, 32): 'a photo of a person kicking a sports ball',
                   (67, 32): 'a photo of a person picking up a sports ball',
                   (81, 32): 'a photo of a person serving a sports ball',
                   (84, 32): 'a photo of a person signing a sports ball',
                   (90, 32): 'a photo of a person spinning a sports ball',
                   (104, 32): 'a photo of a person throwing a sports ball',
                   (57, 32): 'a photo of a person and a sports ball',
                   (36, 11): 'a photo of a person holding a stop sign',
                   (94, 11): 'a photo of a person standing under a stop sign',
                   (97, 11): 'a photo of a person stopping at a stop sign',
                   (57, 11): 'a photo of a person and a stop sign',
                   (8, 28): 'a photo of a person carrying a suitcase',
                   (18, 28): 'a photo of a person dragging a suitcase',
                   (36, 28): 'a photo of a person holding a suitcase',
                   (39, 28): 'a photo of a person hugging a suitcase',
                   (52, 28): 'a photo of a person loading a suitcase',
                   (58, 28): 'a photo of a person opening a suitcase',
                   (60, 28): 'a photo of a person packing a suitcase',
                   (67, 28): 'a photo of a person picking up a suitcase',
                   (116, 28): 'a photo of a person zipping a suitcase',
                   (57, 28): 'a photo of a person and a suitcase',
                   (8, 37): 'a photo of a person carrying a surfboard',
                   (18, 37): 'a photo of a person dragging a surfboard',
                   (36, 37): 'a photo of a person holding a surfboard',
                   (41, 37): 'a photo of a person inspecting a surfboard',
                   (43, 37): 'a photo of a person jumping a surfboard',
                   (49, 37): 'a photo of a person lying on a surfboard',
                   (52, 37): 'a photo of a person loading a surfboard',
                   (76, 37): 'a photo of a person riding a surfboard',
                   (93, 37): 'a photo of a person standing on a surfboard',
                   (87, 37): 'a photo of a person sitting on a surfboard',
                   (111, 37): 'a photo of a person washing a surfboard',
                   (57, 37): 'a photo of a person and a surfboard',
                   (8, 77): 'a photo of a person carrying a teddy bear',
                   (36, 77): 'a photo of a person holding a teddy bear',
                   (39, 77): 'a photo of a person hugging a teddy bear',
                   (45, 77): 'a photo of a person kissing a teddy bear',
                   (57, 77): 'a photo of a person and a teddy bear',
                   (8, 38): 'a photo of a person carrying a tennis racket',
                   (36, 38): 'a photo of a person holding a tennis racket',
                   (41, 38): 'a photo of a person inspecting a tennis racket',
                   (99, 38): 'a photo of a person swinging a tennis racket',
                   (57, 38): 'a photo of a person and a tennis racket',
                   (0, 27): 'a photo of a person adjusting a tie', (15, 27): 'a photo of a person cutting a tie',
                   (36, 27): 'a photo of a person holding a tie', (41, 27): 'a photo of a person inspecting a tie',
                   (70, 27): 'a photo of a person pulling a tie', (105, 27): 'a photo of a person tying a tie',
                   (114, 27): 'a photo of a person wearing a tie', (57, 27): 'a photo of a person and a tie',
                   (36, 70): 'a photo of a person holding a toaster',
                   (59, 70): 'a photo of a person operating a toaster',
                   (75, 70): 'a photo of a person repairing a toaster',
                   (57, 70): 'a photo of a person and a toaster', (12, 61): 'a photo of a person cleaning a toilet',
                   (29, 61): 'a photo of a person flushing a toilet',
                   (58, 61): 'a photo of a person opening a toilet',
                   (75, 61): 'a photo of a person repairing a toilet',
                   (87, 61): 'a photo of a person sitting on a toilet',
                   (93, 61): 'a photo of a person standing on a toilet',
                   (111, 61): 'a photo of a person washing a toilet', (57, 61): 'a photo of a person and a toilet',
                   (6, 79): 'a photo of a person brushing with a toothbrush',
                   (36, 79): 'a photo of a person holding a toothbrush',
                   (111, 79): 'a photo of a person washing a toothbrush',
                   (57, 79): 'a photo of a person and a toothbrush',
                   (42, 9): 'a photo of a person installing a traffic light',
                   (75, 9): 'a photo of a person repairing a traffic light',
                   (94, 9): 'a photo of a person standing under a traffic light',
                   (97, 9): 'a photo of a person stopping at a traffic light',
                   (57, 9): 'a photo of a person and a traffic light',
                   (17, 7): 'a photo of a person directing a truck', (21, 7): 'a photo of a person driving a truck',
                   (41, 7): 'a photo of a person inspecting a truck',
                   (52, 7): 'a photo of a person loading a truck', (75, 7): 'a photo of a person repairing a truck',
                   (76, 7): 'a photo of a person riding a truck', (87, 7): 'a photo of a person sitting on a truck',
                   (111, 7): 'a photo of a person washing a truck', (57, 7): 'a photo of a person and a truck',
                   (8, 25): 'a photo of a person carrying a umbrella',
                   (36, 25): 'a photo of a person holding a umbrella',
                   (53, 25): 'a photo of a person losing a umbrella',
                   (58, 25): 'a photo of a person opening a umbrella',
                   (75, 25): 'a photo of a person repairing a umbrella',
                   (82, 25): 'a photo of a person setting a umbrella',
                   (94, 25): 'a photo of a person standing under a umbrella',
                   (57, 25): 'a photo of a person and a umbrella', (36, 75): 'a photo of a person holding a vase',
                   (54, 75): 'a photo of a person making a vase', (61, 75): 'a photo of a person painting a vase',
                   (57, 75): 'a photo of a person and a vase', (27, 40): 'a photo of a person filling a wine glass',
                   (36, 40): 'a photo of a person holding a wine glass',
                   (85, 40): 'a photo of a person sipping a wine glass',
                   (106, 40): 'a photo of a person toasting a wine glass',
                   (48, 40): 'a photo of a person licking a wine glass',
                   (111, 40): 'a photo of a person washing a wine glass',
                   (57, 40): 'a photo of a person and a wine glass',
                   (26, 22): 'a photo of a person feeding a zebra', (36, 22): 'a photo of a person holding a zebra',
                   (65, 22): 'a photo of a person petting a zebra',
                   (112, 22): 'a photo of a person watching a zebra', (57, 22): 'a photo of a person and a zebra'}

class InputGate(nn.Module):
    def __init__(self, input_size):
        super(InputGate, self).__init__()
        self.control_gate = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        control = self.control_gate(input)
        return control

class GEN(nn.Module):

    def __init__(self, args,d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_inter_dec_layrs=3,
                 return_intermediate_dec=False, num_queries=64, clip_dim=768, enable_cp=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, enable_cp)
        encoder_norm = LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before, False)
        instance_decoder_norm = LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before, False)
        interaction_decoder_norm = LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)

        clip_interaction_decoder_layer = TransformerDecoderFusionLayer(clip_dim, nhead, dim_feedforward,
                                                                       dropout, activation, normalize_before, enable_cp)
        clip_interaction_decoder_norm = LayerNorm(clip_dim)
        self.clip_interaction_decoder = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_inter_dec_layrs,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.inter_guided_embedd = nn.Embedding(num_queries, clip_dim)
        self.queries2spacial_proj = nn.Linear(d_model, clip_dim)
        self.queries2spacial_proj_norm = LayerNorm(clip_dim)

        self.obj_class_fc = nn.Linear(d_model, clip_dim)
        self.obj_class_ln = LayerNorm(clip_dim)

        self.hoi_cls = None

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.return_intermediate = return_intermediate_dec

        self.args=args

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.inter_guided_embedd.weight)

    def build_mae_decoder(self, image_size, device, channel0=2048):
        # Generate spatial shape according to image size
        h, w, c = math.ceil(image_size[0] / 32), math.ceil(image_size[1] / 32), channel0
        total_spatial_shapes = [h, w, c]
        # total_spatial_shapes.append([h, w, c])

        # Build mae decoder
        if self.args.control[0]=='sum':
        #     self.mae_decoder = TransformerDecoderMAE(
        #     hidden_dim=self.d_model,
        #     feedforward_dim=self.dim_feedforward,
        #     num_heads=self.nhead,
        #     dropout=self.dropout,
        #     activation=self.activation,
        #     return_intermediate=self.return_intermediate,
        #     total_spatial_shapes=total_spatial_shapes,
        # )
            self.mae_decoder = TransformerEncoderMAEconcate(
                hidden_dim=self.d_model,
                feedforward_dim=self.dim_feedforward,
                num_heads=self.nhead,
                dropout=self.dropout,
                activation=self.activation,
                return_intermediate=self.return_intermediate,
                total_spatial_shapes=total_spatial_shapes,
            )
        elif self.args.control[0]=='concate':
            self.mae_decoder = TransformerDecoderMAEconcate(
            hidden_dim=self.d_model,
            feedforward_dim=self.dim_feedforward,
            num_heads=self.nhead,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate=self.return_intermediate,
            total_spatial_shapes=total_spatial_shapes,
        )
        elif self.args.control[0]=='query':
            self.mae_decoder = TransformerDecoderMAEquery(
            hidden_dim=self.d_model,
            feedforward_dim=self.dim_feedforward,
            num_heads=self.nhead,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate=self.return_intermediate,
            total_spatial_shapes=total_spatial_shapes,
        )
        elif self.args.control[0]=='encoder_sum':
            self.mae_decoder = TransformerEncoderMAE(
            hidden_dim=self.d_model,
            feedforward_dim=self.dim_feedforward,
            num_heads=self.nhead,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate=self.return_intermediate,
            total_spatial_shapes=total_spatial_shapes,
        )
        elif self.args.control[0]=='encoder_concate':
            self.mae_decoder = TransformerEncoderMAEconcate(
            hidden_dim=self.d_model,
            feedforward_dim=self.dim_feedforward,
            num_heads=self.nhead,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate=self.return_intermediate,
            total_spatial_shapes=total_spatial_shapes,
        )
        else:
            self.mae_decoder = TransformerDecoderMAE(
            hidden_dim=self.d_model,
            feedforward_dim=self.dim_feedforward,
            num_heads=self.nhead,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate=self.return_intermediate,
            total_spatial_shapes=total_spatial_shapes,
        )
        # self.mae_decoder = TransformerDecoderMAE(
        #     hidden_dim=self.d_model,
        #     feedforward_dim=self.dim_feedforward,
        #     num_heads=self.nhead,
        #     dropout=self.dropout,
        #     activation=self.activation,
        #     return_intermediate=self.return_intermediate,
        #     total_spatial_shapes=total_spatial_shapes,
        # )
        self.mae_decoder.to(device)
        # self.gate1=InputGate((256))
        self.gate=Condition(hidden_dim=self.d_model,
            feedforward_dim=self.dim_feedforward,
            num_heads=self.nhead,
            dropout=self.dropout,
            activation=self.activation,
            num_layers=1,
            return_intermediate=self.return_intermediate)
        self.gate.to(device)
        
        # self.gate3=Condition_encoder(hidden_dim=self.d_model,
        #     feedforward_dim=self.dim_feedforward,
        #     num_heads=self.nhead,
        #     dropout=self.dropout,
        #     activation=self.activation,
        #     return_intermediate=self.return_intermediate)
        # self.gate3.to(device)
        
        # self.gate1=InputGate((256))
        # self.gate2=nn.MultiheadAttention(256, 1, dropout=0)
        # self.gate2.to(device)
        # self.mlp=nn.Sequential(
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.mlp.to(device)
    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_model, clip_proj,
                clip_src,enable_mae=False,target=None):
        # flatten NxCxHxW to HWxNxC
        # print(src.shape)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.flatten(1)
        ins_tgt = torch.zeros_like(ins_query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        memory_oral=memory
        if enable_mae:
            # assert self.mae_decoder is not None
            # mae_output = self.mae_decoder(None, memory, mask_flatten=mask)
            # return mae_output
            assert self.mae_decoder is not None
            if self.args.dataset_file=='hico':
                feat=torch.load('genxing_fea.pt', map_location='cpu')
                # feat=torch.load('fea.pt', map_location='cpu')
                text_label_dict=list(hico_text_label.keys())
            else:
                feat=torch.load('fea_vcoco.pt', map_location='cpu')
                text_label_dict=list(vcoco_hoi_text_label.keys())
            ans=[]
            for i in target:
                ans.append(text_label_dict.index((torch.argmax(i['verb_labels'],dim=1),int(i['obj_labels']))))
            # print(ans)
            if self.args.dataset_file=='hico':
                myfeat=torch.cat([feat[i+1].to(memory.device) for i in ans],dim=0)
            else:
                myfeat=torch.cat([feat[i].to(memory.device) for i in ans],dim=0)
            # myfeat=torch.cat([feat[i+1].to(memory.device) for i in ans],dim=0)
            # int_memory=memory
            if self.args.control[0]=='sum':
                # myfeat_gate=self.gate(myfeat)
                # mask_my= myfeat_gate<0.5
                # myfeat = myfeat * mask_my
                # int_memory=memory+0.01*myfeat.permute(1,0,2)
                
                # mae_output = self.mae_decoder(None, int_memory, mask_flatten=mask,mask_my=mask)
                myfeat_gate=self.gate1(myfeat).squeeze(2)
                mask_my= myfeat_gate<0.5
                myfeat = myfeat.permute(1,0,2)
                # int_memory=memory+0.01*myfeat.permute(1,0,2)
                
                int_memory=torch.cat([myfeat,memory])
                
                mask_my=torch.cat([mask_my,mask],dim=1)
                mae_output = self.mae_decoder(int_memory, mask_flatten=mask_my)
            elif self.args.control[0]=='concate':
                # print("121212"*34)
                myfeat_gate=self.gate(myfeat).squeeze(2)

                myfeat = myfeat.permute(1,0,2)
                int_memory=torch.cat([myfeat,memory])
                mask_my= myfeat_gate<0.5
                mask_my=torch.cat([mask_my,mask],dim=1)
                # print(mask.shape)
                mae_output = self.mae_decoder(None, int_memory, mask_flatten=mask,mask_my=mask_my)
                
                # myfeat=self.gate(myfeat).permute(1,0,2)
                
                # int_memory=torch.cat([myfeat,memory])
            elif self.args.control[0]=='query':
                myfeat_gate=self.gate(myfeat)
                myfeat = (myfeat * myfeat_gate).permute(1,0,2)
                mae_output = self.mae_decoder(None, memory, mask_flatten=mask,tgt_my=myfeat)
            elif self.args.control[0]=='encoder_sum':
                # myfeat_gate=self.gate(myfeat)
                # mask_my= myfeat_gate<0.5
                # myfeat = myfeat * mask_my
                # int_memory=memory+myfeat.permute(1,0,2)
                
                # mae_output = self.mae_decoder(int_memory, mask_flatten=mask)

                # 融合方式：逐元素相乘
                # ---------------------------------------------------------------------------
                # ---------------------------------------------------------------------------
                # myfeat_gate=self.gate(myfeat,memory,mask)
                # myfeat = myfeat * myfeat_gate
                # myfeat = myfeat.permute(1,0,2)
                # int_memory=myfeat * memory
                # ---------------------------------------------------------------------------
                # ---------------------------------------------------------------------------
                # 不用条件控制
                # ---------------------------------------------------------------------------
                # ---------------------------------------------------------------------------
                # 逐元素相加
                myfeat_gate=self.gate(myfeat,memory,mask)
                mask_my= myfeat_gate<0.5
                myfeat = myfeat * mask_my
                # myfeat = myfeat * myfeat_gate
                myfeat = myfeat.permute(1,0,2)
                int_memory=myfeat + memory
                # ---------------------------------------------------------------------------
                # ---------------------------------------------------------------------------

                mae_output = self.mae_decoder(int_memory, mask_flatten=mask)
            elif self.args.control[0]=='encoder_concate':
                # 正常的encoder_concate
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                myfeat_gate=self.gate(myfeat,memory,mask).squeeze(2)
                # myfeat_gate=self.gate3(myfeat,memory,mask)[0].squeeze(1)
                mask_my= myfeat_gate<0.5
                myfeat = myfeat.permute(1,0,2)
                # int_memory=memory+0.01*myfeat.permute(1,0,2)
                
                int_memory=torch.cat([myfeat,memory])
                
                mask_my=torch.cat([mask_my,mask],dim=1)
                mae_output = self.mae_decoder(int_memory, mask_flatten=mask_my)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # # 注意力
                # # nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                # myfeat_gate=self.gate2(myfeat.transpose(0,1),memory,value=memory, key_padding_mask=mask)[0].transpose(0,1)
                # myfeat_gate_1=self.mlp(myfeat_gate).squeeze(2)
                # mask_my= myfeat_gate_1<0.5
                # myfeat = myfeat.permute(1,0,2)
                # # int_memory=memory+0.01*myfeat.permute(1,0,2)
                
                # int_memory=torch.cat([myfeat,memory])
                
                # mask_my=torch.cat([mask_my,mask],dim=1)
                # mae_output = self.mae_decoder(int_memory, mask_flatten=mask_my)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            else:
                int_memory=memory
            # print(mask.shape)
                mae_output = self.mae_decoder(None, int_memory, mask_flatten=mask,mask_my=mask)
            # print(mae_output[0].shape)
            # print(len(mae_output))
            return mae_output

        ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        # original
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs_ori = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                         pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs_ori = inter_hs_ori.transpose(1, 2)
        memory = self.obj_class_ln(self.obj_class_fc(memory))

        # h_hs_detached = h_hs.detach()

        inter_hs = (h_hs + o_hs) / 2.0
        inter_hs = self.queries2spacial_proj(inter_hs[-1])
        inter_hs = self.queries2spacial_proj_norm(inter_hs)
        # inter_hs = inter_hs + self.inter_guided_embedd.weight.unsqueeze(0).repeat(bs, 1, 1)

        dtype = inter_hs.dtype

        clip_cls_feature, clip_visual = clip_model.encode_image(clip_src)
        clip_cls_feature = clip_cls_feature / clip_cls_feature.norm(dim=1, keepdim=True)
        clip_cls_feature = clip_cls_feature.to(dtype)
        clip_visual = clip_visual.to(dtype)
        with torch.no_grad():
            clip_hoi_score = clip_cls_feature @ self.hoi_cls.T
            # obj_score = clip_cls_feature @ self.obj_cls.T
            # obj_hoi_score = obj_score @ self.obj2hoi_proj

            # verb_score = clip_cls_feature @ self.verb_cls.T
            # verb_hoi_score = verb_score @ self.verb2hoi_proj
            # clip_hoi_score += verb_hoi_score * 0.1
            # ignore_idx = clip_hoi_score.sort(descending=True).indices[:, self.topk:]
            # for idx, igx in enumerate(ignore_idx):
            #     clip_hoi_score[idx][igx] *= 0
            clip_hoi_score = clip_hoi_score.unsqueeze(1)

        clip_cls_feature = clip_cls_feature.unsqueeze(1).repeat(1, num_queries, 1)

        inter_hs = self.clip_interaction_decoder(inter_hs.permute(1, 0, 2),
                                                 clip_visual.permute(1, 0, 2), sup_memory=memory)
        inter_hs_oral=inter_hs
        inter_hs_oral=inter_hs_oral.transpose(1,2)
        inter_hs = inter_hs @ clip_proj.to(dtype)
        inter_hs = inter_hs.permute(0, 2, 1, 3)
        # print(inter_hs.shape)

        # add
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        # #torch.Size([3, 64, 8, 256])
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                     pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs = inter_hs.transpose(1, 2)
        return h_hs, o_hs, inter_hs, memory_oral.permute(1, 0, 2).view(bs, 1, h*w, c),inter_hs_oral[-1],clip_cls_feature, clip_hoi_score, clip_visual @ clip_proj.to(dtype)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderCLIP(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(output.shape) == 4:
                output = output[i]
            else:
                # only this branch will be used, we only use last human/object query and pass one layer decoder block
                output = output
            output = layer(output, memory, sup_memory=sup_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.enable_cp:
            def _inner_forward(args):
                src_inner, q_inner, k_inner, src_mask_inner, src_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=src_inner, attn_mask=src_mask_inner,
                                      key_padding_mask=src_key_padding_mask_inner)[0]
                return src_inner

            src2 = cp.checkpoint(_inner_forward, (src, q, k, src_mask, src_key_padding_mask))
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, sup_memory=None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(sup_memory, pos),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt3 = tgt + self.dropout4(tgt3)
        # tgt3 = self.norm4(tgt3)
        tgt3 = tgt + self.dropout2(tgt3)
        tgt3 = self.norm2(tgt3)

        tgt = tgt2 + tgt3

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, sup_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderMAE(TransformerDecoder):

    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True,
                 total_spatial_shapes=None):
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes

        decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        super(TransformerDecoderMAE, self).__init__(decoder_layer, num_layers, decoder_norm, return_intermediate)
        self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_embed_list = nn.Embedding(total_spatial_shapes[0] * total_spatial_shapes[1], hidden_dim * 2)
        self.pos = nn.Embedding(total_spatial_shapes[0] * total_spatial_shapes[1], hidden_dim)
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        a=1024*3
        self.output_proj = nn.Linear(hidden_dim, a)

    def forward(self, tgt, src, mask_flatten=None,mask_my=None):
        bs = src.shape[1]
        mae_output = []

        query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        query_pos = query_pos.unsqueeze(1).expand(-1, bs, -1)
        tgt = tgt.unsqueeze(1).expand(-1, bs, -1)
        pos = self.pos.weight.unsqueeze(1).expand(-1, bs, -1)
        tgt_mask = mask_flatten.transpose(0, 1).unsqueeze(-1)
        h, w, c = self.spatial_shapes[0], self.spatial_shapes[1], self.spatial_shapes[2]
        tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight. \
            expand(h * w, bs, -1) * tgt_mask.to(tgt.dtype)

        hs = super(TransformerDecoderMAE, self).forward(
            tgt, src, memory_key_padding_mask=mask_my, pos=pos, query_pos=query_pos
        )
        hs = hs.transpose(1, 2)
        output = self.output_proj(hs)

        mae_output.append(output[-1].transpose(-2, -1).reshape(-1, c, h, w))
        return mae_output

class TransformerDecoderMAEconcate(TransformerDecoder):

    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True,
                 total_spatial_shapes=None):
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
       
        decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        super(TransformerDecoderMAEconcate, self).__init__(decoder_layer, num_layers,decoder_norm, return_intermediate)
        self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_embed_list = nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim * 2)
        self.pos=nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim)
        a=1024*3
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        self.output_proj = nn.Linear(hidden_dim, a)

    def forward(self, tgt, src, mask_flatten=None,mask_my=None):
        bs = src.shape[1]
        mae_output = []
        
        query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        query_pos = query_pos.unsqueeze(1).expand(-1,bs,-1)
        tgt = tgt.unsqueeze(1).expand(-1,bs,-1)
        pos=self.pos.weight.unsqueeze(1).expand(-1,bs,-1)
        tgt_mask = mask_flatten.transpose(0,1).unsqueeze(-1)
        h, w, c = self.spatial_shapes[0],self.spatial_shapes[1],self.spatial_shapes[2]
        tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight.\
            expand(h * w, bs,-1) * tgt_mask.to(tgt.dtype)
        # print(pos.shape)
        # print(mask_my.shape)
        # print(src.shape)
        hs = super(TransformerDecoderMAEconcate, self).forward(
            tgt, src,memory_key_padding_mask=mask_my,pos=pos,query_pos=query_pos
        )
        hs = hs.transpose(1,2)
        output = self.output_proj(hs)
        # print(hs.shape)
        output = self.output_proj(hs)
        # print("output:",output.shape)
        # print(h,w,c)
        
        mae_output.append(output[-1].transpose(-2, -1).reshape(-1, c, h, w))
        return mae_output

class TransformerDecoderMAEquery(TransformerDecoder):

    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True,
                 total_spatial_shapes=None):
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
       
        decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        super(TransformerDecoderMAEquery, self).__init__(decoder_layer, num_layers,decoder_norm, return_intermediate)
        self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_embed_list = nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim * 2)
        self.pos=nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim)
        a=1024*3
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        self.output_proj = nn.Linear(hidden_dim, a)

    def forward(self, tgt, src, mask_flatten=None,tgt_my=None):
        bs = src.shape[1]
        mae_output = []
        
        query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        query_pos = query_pos.unsqueeze(1).expand(-1,bs,-1)
        tgt = tgt.unsqueeze(1).expand(-1,bs,-1)
        pos=self.pos.weight.unsqueeze(1).expand(-1,bs,-1)
        tgt_mask = mask_flatten.transpose(0,1).unsqueeze(-1)
        h, w, c = self.spatial_shapes[0],self.spatial_shapes[1],self.spatial_shapes[2]
        tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight.\
            expand(h * w, bs,-1) * tgt_mask.to(tgt.dtype)
        tgt+=tgt_my
        hs = super(TransformerDecoderMAEquery, self).forward(
            tgt, src,memory_key_padding_mask=mask_flatten,pos=pos,query_pos=query_pos
        )
        hs = hs.transpose(1,2)
        output = self.output_proj(hs)
        
        mae_output.append(output[-1].transpose(-2, -1).reshape(-1, c, h, w))
        return mae_output


class TransformerEncoderMAE(TransformerEncoder):
    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True,
                 total_spatial_shapes=None):
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
       
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim)
        super(TransformerEncoderMAE, self).__init__(encoder_layer, num_layers,encoder_norm)
        self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_embed_list = nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim * 2)
        self.pos=nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim)
        a=1024*3
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        self.output_proj = nn.Linear(hidden_dim, a)

    def forward(self, tgt,mask_flatten=None):
        bs = tgt.shape[1]
        mae_output = []
        
        # query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        # query_pos = query_pos.unsqueeze(1).expand(-1,bs,-1)
        # tgt = tgt.unsqueeze(1).expand(-1,bs,-1)
        pos=self.pos.weight.unsqueeze(1).expand(-1,bs,-1)
        tgt_mask = mask_flatten.transpose(0,1).unsqueeze(-1)
        h, w, c = self.spatial_shapes[0],self.spatial_shapes[1],self.spatial_shapes[2]
        tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight.\
            expand(h * w, bs,-1) * tgt_mask.to(tgt.dtype)
        
        hs = super(TransformerEncoderMAE, self).forward(
            tgt, src_key_padding_mask=mask_flatten,pos=pos
        )
        # hs = hs.transpose(1,2)
        # print(hs.shape)
        output = self.output_proj(hs)
        # print("output:",output.shape)
        # print(h,w,c)
        # mae_output.append(output.permute(1,2,0).reshape(-1, c, h, w))
        mae_output.append(output.permute(1,2,0))
        # print(mae_output[0].shape)
        return mae_output

class TransformerEncoderMAEconcate(TransformerEncoder):
    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True,
                 total_spatial_shapes=None):
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
       
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim)
        super(TransformerEncoderMAEconcate, self).__init__(encoder_layer, num_layers,encoder_norm)
        self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_embed_list = nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim * 2)
        self.pos=nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1]*2, hidden_dim)
        a=1024*3
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        self.output_proj = nn.Linear(hidden_dim, a)

    def forward(self, tgt,mask_flatten=None):
        bs = tgt.shape[1]
        mae_output = []
        
        # query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        # query_pos = query_pos.unsqueeze(1).expand(-1,bs,-1)
        # tgt = tgt.unsqueeze(1).expand(-1,bs,-1)
        pos=self.pos.weight.unsqueeze(1).expand(-1,bs,-1)
        tgt_mask = mask_flatten.transpose(0,1).unsqueeze(-1)
        h, w, c = self.spatial_shapes[0],self.spatial_shapes[1],self.spatial_shapes[2]
        # print(tgt.shape)
        # print(tgt_mask.shape)
        # print(self.mask_query.weight.shape)
        tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight.\
            expand(h * w*2, bs,-1) * tgt_mask.to(tgt.dtype)
        
        hs = super(TransformerEncoderMAEconcate, self).forward(
            tgt, src_key_padding_mask=mask_flatten,pos=pos
        )
        # hs = hs.transpose(1,2)
        # print(hs.shape)
        hs=hs[256:,:,:]
        output = self.output_proj(hs)
        # print("output:",output.shape)
        # print(h,w,c)
        # print(output.shape)
        ddd=output.permute(1,2,0)
        # print(ddd.shape)
        mae_output.append(ddd)
        # print(mae_output[0].shape)
        return mae_output



class Condition(TransformerDecoder):

    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True):
        # total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes

        decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)
        super(Condition, self).__init__(decoder_layer, num_layers, decoder_norm, return_intermediate)
        # self.spatial_shapes = total_spatial_shapes
        # self.hidden_dim = hidden_dim
        # self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_pos = nn.Embedding(256, hidden_dim)
        self.pos = nn.Embedding(256, hidden_dim)
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        # a=1024*3
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, tgt, src, mask_flatten=None):
        bs = src.shape[1]
        mae_output = []

        query_pos = self.query_pos.weight.unsqueeze(1).expand(-1, bs, -1)
        # tgt = tgt.unsqueeze(1).expand(-1, bs, -1)
        pos = self.pos.weight.unsqueeze(1).expand(-1, bs, -1)
        # tgt_mask = mask_flatten.transpose(0, 1).unsqueeze(-1)
        # h, w, c = self.spatial_shapes[0], self.spatial_shapes[1], self.spatial_shapes[2]
        # tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight. \
        #     expand(h * w, bs, -1) * tgt_mask.to(tgt.dtype)

        hs = super(Condition, self).forward(
            tgt.transpose(0, 1), src, memory_key_padding_mask=mask_flatten, pos=pos, query_pos=query_pos
        )
        hs = hs.transpose(1, 2)
        output = self.output_proj(hs)
        output=self.sigmoid(output)
        # mae_output.append(output[-1])
        return output[-1]

class Condition_encoder(TransformerEncoder):
    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=True):
        # total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
       
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim)
        super(Condition_encoder, self).__init__(encoder_layer, num_layers,encoder_norm)
        # self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        # self.mask_query = nn.Embedding(1, hidden_dim)
        # self.query_embed_list = nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim * 2)
        self.pos=nn.Embedding(512, hidden_dim)
        # a=1024*3
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, tgt,src,mask_flatten=None):
        query=torch.cat((tgt.transpose(0,1),src))
        tt=torch.zeros((2, 256),device=tgt.device)
        mask=torch.cat((tt,mask_flatten),dim=1)
        bs = query.shape[1]
        mae_output = []
        
        # query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        # query_pos = query_pos.unsqueeze(1).expand(-1,bs,-1)
        # tgt = tgt.unsqueeze(1).expand(-1,bs,-1)
        pos=self.pos.weight.unsqueeze(1).expand(-1,bs,-1)
        # tgt_mask = mask_flatten.transpose(0,1).unsqueeze(-1)
        # h, w, c = self.spatial_shapes[0],self.spatial_shapes[1],self.spatial_shapes[2]
        # # print(tgt.shape)
        # # print(tgt_mask.shape)
        # # print(self.mask_query.weight.shape)
        # tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight.\
        #     expand(h * w*2, bs,-1) * tgt_mask.to(tgt.dtype)
        
        hs = super(Condition_encoder, self).forward(
            query, src_key_padding_mask=mask,pos=pos
        )
        # hs = hs.transpose(1,2)
        # print(hs.shape)
        hs=hs[256:,:,:]
        output = self.output_proj(hs)
        # print("output:",output.shape)
        # print(h,w,c)
        # print(output.shape)
        ddd=output.permute(1,2,0)
        # print(ddd.shape)
        mae_output.append(ddd)
        # print(mae_output[0].shape)
        return mae_output



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        num_inter_dec_layrs=args.inter_dec_layers,
        args=args
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
