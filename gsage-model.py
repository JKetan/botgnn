import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE

class BotGraphSAGE(nn.Module):
    def __init__(self, des_size=768, tweet_size = 768, num_prop_size=4, cat_prop_size=3, embedding_dimension=128):
        super(BotGraphSAGE, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        # self.linear_relu_tweet=nn.Sequential(
        #     nn.Linear(tweet_size,int(embedding_dimension/4)),
        #     nn.LeakyReLU()
        # )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 3)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 3)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = GraphSAGE(in_channels=embedding_dimension, out_channels=embedding_dimension, hidden_channels=64, num_layers=2)
        self.sage2 = GraphSAGE(in_channels=embedding_dimension, out_channels=embedding_dimension, hidden_channels=64, num_layers=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index):
        d = self.linear_relu_des(des)
        # t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        #x = c
        x = torch.cat((d, n, c), dim=1)

        x = self.linear_relu_input(x)

        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = self.sage2(x, edge_index)
        x = torch.relu(x)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x
    
class BotGraphSAGE1(nn.Module):
    def __init__(self, des_size=768, tweet_size = 768, num_prop_size=4, cat_prop_size=3, embedding_dimension=128):
        super(BotGraphSAGE, self).__init__()
        # self.linear_relu_des = nn.Sequential(
        #     nn.Linear(des_size, int(embedding_dimension / 4)),
        #     nn.LeakyReLU()
        # )
        # self.linear_relu_tweet=nn.Sequential(
        #     nn.Linear(tweet_size,int(embedding_dimension/4)),
        #     nn.LeakyReLU()
        # )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = GraphSAGE(in_channels=embedding_dimension, out_channels=embedding_dimension, hidden_channels=64, num_layers=2)
        self.sage2 = GraphSAGE(in_channels=embedding_dimension, out_channels=embedding_dimension, hidden_channels=64, num_layers=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index):
        # d = self.linear_relu_des(des)
        # t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        #x = c
        x = torch.cat((n, c), dim=1)

        x = self.linear_relu_input(x)

        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = self.sage2(x, edge_index)
        x = torch.relu(x)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x
    
class BotGraphSAGE2(nn.Module):
    def __init__(self, des_size=768, tweet_size = 768, num_prop_size=4, cat_prop_size=3, embedding_dimension=128):
        super(BotGraphSAGE, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = GraphSAGE(in_channels=embedding_dimension, out_channels=embedding_dimension, hidden_channels=64, num_layers=2)
        self.sage2 = GraphSAGE(in_channels=embedding_dimension, out_channels=embedding_dimension, hidden_channels=64, num_layers=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        #x = c
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)

        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = self.sage2(x, edge_index)
        x = torch.relu(x)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x