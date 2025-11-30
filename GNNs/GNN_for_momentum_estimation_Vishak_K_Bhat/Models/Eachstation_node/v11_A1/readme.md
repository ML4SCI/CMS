## The model architecture

    MODEL_GNN(
    (conv1): MPL()
    (conv2): MPL()
    (conv3): MPL()
    (conv4): MPL()
    (lin1): Linear(in_features=128, out_features=128, bias=True)
    (lin2): Linear(in_features=128, out_features=16, bias=True)
    (lin3): Linear(in_features=16, out_features=16, bias=True)
    (lin4): Linear(in_features=16, out_features=1, bias=True)
    (lin5): Linear(in_features=128, out_features=128, bias=True)
    (lin6): Linear(in_features=128, out_features=16, bias=True)
    (lin7): Linear(in_features=16, out_features=16, bias=True)
    (lin8): Linear(in_features=16, out_features=1, bias=True)
    (global_att_pool1): GlobalAttention(gate_nn=Sequential(
      (0): Linear(in_features=64, out_features=1, bias=True)
    ), nn=None)
    (global_att_pool2): GlobalAttention(gate_nn=Sequential(
      (0): Linear(in_features=64, out_features=1, bias=True)
    ), nn=None)
    )

## Training HyperParamters



```bibtext
criterion- 
def CustompTLoss(output, target, lower_pt_limit):
    
    if not isinstance(lower_pt_limit, torch.Tensor):
        lower_pt_limit = torch.tensor(lower_pt_limit)
    
    lower_pt_limit = lower_pt_limit.to(output.dtype)
    output =torch.clip(output, min=lower_pt_limit.to(device))
    loss = torch.mean((target - output)**2 + torch.gt(output, lower_pt_limit.long() * \
        (1 / (1 + torch.exp(-(output - lower_pt_limit) * 3)) - 1) + \
            torch.le(output, lower_pt_limit).long()*(-1/2)))
    return loss

```


```bibtext 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)
```

```bibtext 
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1, factor=0.5)
```
