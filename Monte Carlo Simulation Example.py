#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Riddler Express
# 
# Santa Claus is getting up there in age, and his memory has begun to falter. (After all, why do you think he keeps a list?) It’s gotten so bad that this year Santa forgot what order to put the reindeer in. Obviously, he remembers that Rudolph goes first because of the red organic light bulb in the middle of his face, but big guy just can’t remember what to do with the other eight.
# 
# If he doesn’t get the right order, the aerodynamics of his sleigh will be all wrong and he won’t be able to get all of his deliveries done in time. Yes, Santa has Moneyballed Christmas Eve. Luckily, the reindeer know where they should each be, but since they’re just animals they can only grunt in approval if they are put in the right spot.
# 
# Determined to get it right, Santa first creates a list of the reindeer in some random order. He then goes to the first position and harnesses each reindeer one by one, starting at the top of his list. When a reindeer grunts, Santa leaves it in that correct position, moves onto the next position, and works down that same list once again.
# 
# If harnessing a reindeer into any spot takes one minute, how long on average would it take Santa to get the correct reindeer placement?
# 
# Extra credit: Is there a strategy that Santa could use that does better?

# In[125]:


x = 1
time_in_minutes = 1
time_list = []
r_order = list(range(1,9))
random.shuffle(r_order)
solution = list(range(1,9))
random.shuffle(solution)
while x < 1000001:
    while solution != []:
        solution = [x for x in solution if x != 0]
        r_order = [x for x in r_order if x != 0]
        for i in range(len(solution)):
            if r_order[i] == solution[i]:
                solution[i] = 0
                r_order[i] = 0
            time_in_minutes += 1
        random.shuffle(r_order)
    time_list.append(time_in_minutes)
    r_order = list(range(1,9))
    random.shuffle(r_order)
    solution = list(range(1,9))
    random.shuffle(solution)
    time_in_minutes = 0
    x += 1
print("The average time it takes Santa is {} minutes.".format(sum(time_list)/len(time_list)))


# In[126]:


plt.hist(time_list, bins = 50)


# In[128]:


solution = list(range(1,9))
random.shuffle(solution)
time_in_min = 1
x = 1
time_list_b = []
while x < 1000001:
    checked = []
    while solution != [0]:
        r_int = random.randint(1,8)
        if r_int not in checked:
            solution = [x for x in solution if x != 0]
            for i in range(len(solution)):
                if solution[i] == r_int:
                    solution[i] = 0
                    checked.append(r_int)
                    time_in_min += 1*(i+1)
                continue
    x += 1
    time_list_b.append(time_in_min)
    time_in_min = 0
    solution = list(range(1,9))
    random.shuffle(solution)
print(sum(time_list_b)/len(time_list_b))


# In[129]:


plt.hist(time_list_b, bins = 50)


# In[ ]:




