---
layout: post
title: "TIL: Why the M4 Competition Winner Combined Old-School Stats with LSTMs"
date: 2024-01-16
categories: forecasting neural-networks
---

I fell down an interesting rabbit hole while looking into forecasting techniques the other day. I kept seeing references to something called the "M4 Competition" from 2018, and what caught my eye was this fascinating detail: the winning solution wasn't just another fancy neural network. Instead, this guy Slawek Smyl basically said "hey, what if we combined classical statistical methods with modern neural nets?"

The more I dug into his solution, the more intrigued I got by how he pulled it off. He created this hybrid approach that used exponential smoothing (old-school stats) together with a really clever neural architecture using dilated LSTMs.

## The Dilated LSTM Part - Why It's Cool

So here's what makes the dilated LSTM component so interesting. In a regular LSTM, you process your sequence step by step, looking at each time point. But with dilation, you introduce these "gaps" in how you look at the sequence. Think of it like this:

- Layer 1 (dilation=1): Looks at every point: [1,2,3,4,5,6]
- Layer 2 (dilation=2): Skips every other: [1,_,3,_,5,_]
- Layer 3 (dilation=4): Bigger gaps: [1,_,_,_,5,_]

This might seem counterintuitive at first (why would skipping data points help?), but it's actually brilliant for time series. The different dilation rates let the model capture patterns at different time scales without needing a super deep network. Lower layers can focus on recent, fine-grained patterns, while higher layers catch longer-term dependencies.

Here's what my ported implementation looks like so far:

```python
class AttentiveDilatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, dilations):
        super().__init__()
        self.dilations = dilations
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size, hidden_size) for _ in dilations
        ])
        self.attention = nn.Linear(hidden_size, attention_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        outputs = []
        
        for i, dilation in enumerate(self.dilations):
            # This is where it gets interesting
            # We need to reshape our input to handle the dilation
            dilated_x = self._apply_dilation(x, dilation)
            lstm_out, _ = self.lstm_layers[i](dilated_x)
            outputs.append(lstm_out)
            
        # Still working on the attention mechanism...
```

The trickiest part I'm working on now is the `_apply_dilation` method. You need to carefully reshape your input sequence to create those gaps, but in a way that maintains the temporal relationships in your data. Plus, you need to make sure the dilated sequences align properly when you combine outputs from different layers.

The attention mechanism is next on my list - it basically helps the model figure out which historical patterns are most relevant for the current prediction. It's like having a smart filter that can say "hey, this pattern from 30 steps ago is really similar to what we're seeing now."

I'm still working through some of the trickier parts of the implementation, especially getting the dilation patterns working correctly. But it's pretty cool to see how combining classical statistical knowledge with modern neural networks can lead to better results than either approach alone.

More implementation details to come as I work through this!

*References:*
- Smyl's winning solution: [https://github.com/Mcompetitions/M4-methods/tree/master/118%20-%20slaweks17]
- Original M4 Competition paper: [https://github.com/Mcompetitions/M4-methods/blob/master/118%20-%20slaweks17/ES_RNN_SlawekSmyl.pdf]
