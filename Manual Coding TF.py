#!/usr/bin/env python
# coding: utf-8

# ## Some intro stuff

# In[32]:


class SimpleClass():
    
    def __init__(self,name):
        print("hello", name)
        
    def yell(self):
        print("YELLING")


# In[33]:


s = "world"


# In[34]:


print(type(s))


# In[35]:


x = SimpleClass('Jose') #prints on initalization


# In[30]:


x #instance of the class


# In[18]:


x.yell()


# In[38]:


class ExtendedClass(SimpleClass):
    
    def __init__(self):
        
        super().__init__('Billy')
        print("Hello!")


# In[39]:


y = ExtendedClass()


# In[24]:


y.yell()


# ## Creating the operations class

# In[3]:


class Operation():
    
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []
        
        for node in input_nodes:
            node.output_nodes.append(self)
            
        _default_graph.operations.append(self)
            
    def compute(self):
        pass


# In[4]:


class add(Operation):
    
    def __init__(self, x, y):
        super().__init__([x,y])
        
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


# In[5]:


class multiply(Operation):
    
    def __init__(self, a, b):
        super().__init__([a,b])
        
    def compute(self, a_var, b_var):
        self.inputs = [a_var, b_var]
        return a_var * b_var


# In[6]:


class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
        
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


# In[7]:


class placeholder():
    
    def __init__(self):
        
        self.output_nodes = []
        
        _default_graph.placeholders.append(self)


# In[8]:


class Variable():
    
    def __init__(self, initial_value = None):
        self.value = initial_value
        self.output_nodes = []
        
        _default_graph.variables.append(self)


# In[9]:


class Graph():
    
    def __init__(self): #will fill in to this place, we will have graph having these values
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        global _default_graph #makes it a global variable so can access in any class
        _default_graph = self


# z = Ax + b 
# A = 10 b = 1
# z = 10x + 1

# In[10]:


g = Graph()


# In[11]:


g.set_as_default()


# In[12]:


A = Variable(10)


# In[13]:


b = Variable(1)


# In[14]:


x = placeholder()


# In[15]:


y = multiply(A, x)


# In[16]:


z = add(y, b)


# In[17]:


def traverse_postorder(operation):
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
        
    recurse(operation)
    return nodes_postorder


# In[43]:


class Session():
    
    def run(self, operation, feed_dict = {}):
        
        node_postorder = traverse_postorder(operation)
        
        for node in node_postorder:
            if type(node) == placeholder:
                
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                
                node.output = node.value
                
            else: #OPERATION
                
                node.inputs = [input_node.output for input_node in node.input_nodes]
                
                node.output = node.compute(*node.inputs)
                
            if type(node.output) == list:
                import numpy as np
                node.output = np.array(node.output)
                
        return operation.output


# In[44]:


sess = Session()


# In[45]:


result = sess.run(operation = z, feed_dict = {x: 10})


# In[46]:


print(result)


# In[47]:


g = Graph()
g.set_as_default()


# In[48]:


A = Variable([[10,20], [30,40]])
b = Variable([1,2])


# In[49]:


x = placeholder()


# In[50]:


y = matmul(A, x)


# In[51]:


z = add(y, b)


# In[52]:


sess = Session()


# In[53]:


sess.run(z, feed_dict = {x:10})


# ## Classification

# In[57]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


def sigmoid(z):
    return 1 / (1+np.exp(-z))


# In[64]:


sample_z = np.linspace(-10,10,100)
sample_a = sigmoid(sample_z)


# In[65]:


plt.plot(sample_z, sample_a)


# In[106]:


class Sigmoid(Operation):
    
    def __init__(self, z):
        
        super().__init__([z])
        
    def compute(self, z_val):
        return 1 / (1+np.exp(-z_val))


# In[107]:


from sklearn.datasets import make_blobs


# In[108]:


data = make_blobs(n_samples = 50, n_features = 2, centers = 2, random_state = 75)


# In[109]:


data


# In[110]:


type(data)


# In[111]:


features = data[0]


# In[112]:


labels = data[1]


# In[113]:


features


# In[114]:


plt.scatter(features[:, 0], features[:, 1], c = labels, cmap = 'coolwarm')


# In[115]:


x = np.linspace(0,11,10)


# In[116]:


y = -x


# In[117]:


plt.scatter(features[:, 0], features[:, 1], c = labels, cmap = 'coolwarm')
plt.plot(y)


# In[118]:


y = -x + 5


# In[119]:


plt.scatter(features[:, 0], features[:, 1], c = labels, cmap = 'coolwarm')
plt.plot(y)


# In[120]:


np.array([1,1]).dot(np.array([[8],[10]])) - 5


# In[121]:


np.array([1,1]).dot(np.array([[2], [-10]])) - 5


# In[122]:


g = Graph()


# In[123]:


g.set_as_default()


# In[124]:


x = placeholder()


# In[125]:


w = Variable([1,1])


# In[126]:


b = Variable(-5)


# In[127]:


z = add(matmul(w,x), b)
a = Sigmoid(z)


# In[128]:


sess = Session()


# In[130]:


sess.run(operation = a, feed_dict = {x: [8,10]})


# In[131]:


sess.run(operation = a, feed_dict = {x: [-2, -10]})


# In[ ]:




