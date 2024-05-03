import node
import connection
import random

class Brain:
    def __init__(self, inputs):
        self.conections = []
        self.nodes = []
        self.inputs = inputs
        self.net = []
        self.layers = 2 

        # create input nodes
        for idx in range(self.inputs):
            self.nodes.append(node.Node(idx))
            self.nodes[idx].layer = 0

        # create bias node
        self.nodes.append(node.Node(3))
        self.nodes[3].layer = 0

        # create output node
        self.nodes.append(node.Node(4))
        self.nodes[4].layer = 1

        # create connections
        for idx in range(4):
            self.conections.append(connection.Connection(
                self.nodes[idx], self.nodes[4], random.uniform(-1, 1)
            )) 

    def connect_nodes(self):
        
