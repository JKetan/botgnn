import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class BotGAT(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=4,cat_prop_size=3,embedding_dimension=128,dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/3)),
            nn.LeakyReLU()
        )
        # self.linear_relu_tweet=nn.Sequential(
        #     nn.Linear(tweet_size,int(embedding_dimension/4)),
        #     nn.LeakyReLU()
        # )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/3)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/3)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/3),heads=3)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index):
        d=self.linear_relu_des(des)
        #t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGAT1(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=4,cat_prop_size=3,embedding_dimension=128,dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        # self.linear_relu_des=nn.Sequential(
        #     nn.Linear(des_size,int(embedding_dimension/3)),
        #     nn.LeakyReLU()
        # )
        # self.linear_relu_tweet=nn.Sequential(
        #     nn.Linear(tweet_size,int(embedding_dimension/4)),
        #     nn.LeakyReLU()
        # )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/3),heads=3)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index):
        #d=self.linear_relu_des(des)
        #t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGAT2(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=4,cat_prop_size=3,embedding_dimension=128,dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/3),heads=3)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x